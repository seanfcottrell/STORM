#!/usr/bin/env python
# -*- coding: utf-8 -*-
#LymphAnalysis
import os

SEED = 0

# must be set before heavy numeric libs start
os.environ["PYTHONHASHSEED"] = str(SEED)

# GPU determinism (if you run on CUDA); safe to set even if CPU
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# make BLAS threading deterministic-ish (optional but helps)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import numpy as np
import torch

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

try:
    torch.use_deterministic_algorithms(True)
except Exception as e:
    print("[warn] torch deterministic algorithms not fully enabled:", repr(e))

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import sys
import pandas as pd
import scanpy as sc
import anndata
import scipy.sparse as sp

from utils import _preprocess_and_hvg
from TensorConstructionUtils import (build_irregular_slices,
    intersect_hvgs_and_align)
from TensorDecompositionUtils import attach_QHD_embeddings
from DeconvolutionUtils import deconvolve_gbdt_from_pseudospots, deconvolve_lle_from_pseudospots, consensus_simplex
from GraphConstructions import ppi_graph,build_L_st_with_pseudo_mnn
from parafac2 import fit_parafac2_graph
from PseudoSpotUtils import pseudo_spot_generation

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score

import numpy as np
import pandas as pd
import mygene


def map_ensembl_gene_list_to_symbols(
    genes,
    *,
    species="human",
    batch_size=1000,
    keep_unmapped="original",   # "original", "na", or a literal string like "__NA__"
    strict_duplicates=False,
    verbose=True,
):
    """
    Map a list/Index of genes (typically Ensembl IDs) to gene symbols, while
    preserving the original order and length.

    Parameters
    ----------
    genes : sequence-like
        Gene IDs in the order used by your aligned matrices.
    species : str or int
        Passed to MyGeneInfo.querymany (e.g. "human" or 9606).
    batch_size : int
        Batch size for MyGene queries.
    keep_unmapped : {"original", "na"} or str
        What to do for genes that do not map:
          - "original": keep the cleaned original ID (recommended; length/order preserved)
          - "na": use pd.NA
          - any other string: use that literal fallback string
    strict_duplicates : bool
        If True, raise an error when multiple genes map to the same symbol.
        This can matter if your ppi_graph expects unique node names.
    verbose : bool
        Print summary info.

    Returns
    -------
    mapped_genes : list[str or pd.NA]
        Same length and same order as input `genes`.
    mapping_df : pd.DataFrame
        Columns:
          - original_gene
          - cleaned_gene
          - mapped_symbol_raw
          - gene_for_ppi
    """
    genes = pd.Index(pd.Series(genes, dtype="object").astype(str))
    genes_clean = genes.str.replace(r"\.\d+$", "", regex=True)

    # Detect Ensembl-like IDs; leave non-Ensembl entries unchanged
    ens_mask = genes_clean.str.match(r"^ENS[A-Z]*G\d+$", na=False)
    ens_unique = pd.Index(genes_clean[ens_mask].unique())

    symbol_map = {}

    if len(ens_unique) > 0:
        mg = mygene.MyGeneInfo()
        chunks = []

        for i in range(0, len(ens_unique), batch_size):
            chunk = ens_unique[i:i + batch_size].tolist()
            res = mg.querymany(
                chunk,
                scopes="ensembl.gene",
                fields=["symbol", "name"],
                species=species,
                as_dataframe=True,
                returnall=False,
                verbose=False,
            )

            if isinstance(res, pd.DataFrame):
                res = res.copy()

                # remove explicit notfound rows if present
                if "notfound" in res.columns:
                    res = res[~res["notfound"].fillna(False)]

                # if MyGene returns multiple rows for one query, keep first
                res = res[~res.index.duplicated(keep="first")]
                chunks.append(res)

        if len(chunks) > 0:
            hits = pd.concat(chunks, axis=0)
            hits = hits[~hits.index.duplicated(keep="first")]
            hits = hits.reindex(ens_unique)

            if "symbol" in hits.columns:
                raw_symbols = hits["symbol"]
            else:
                raw_symbols = pd.Series(index=ens_unique, dtype=object)

            for g, s in raw_symbols.items():
                if pd.isna(s) or str(s).strip() == "":
                    symbol_map[g] = pd.NA
                else:
                    symbol_map[g] = str(s)

    mapped_raw = []
    mapped_final = []

    for orig, clean, is_ens in zip(genes.tolist(), genes_clean.tolist(), ens_mask.tolist()):
        if is_ens:
            sym = symbol_map.get(clean, pd.NA)
        else:
            # already looks like a symbol or other non-Ensembl gene name
            sym = clean

        mapped_raw.append(sym)

        if pd.isna(sym):
            if keep_unmapped == "original":
                mapped_final.append(clean)
            elif keep_unmapped == "na":
                mapped_final.append(pd.NA)
            else:
                mapped_final.append(str(keep_unmapped))
        else:
            mapped_final.append(sym)

    mapping_df = pd.DataFrame({
        "original_gene": genes.tolist(),
        "cleaned_gene": genes_clean.tolist(),
        "mapped_symbol_raw": mapped_raw,
        "gene_for_ppi": mapped_final,
    })

    # duplicate check after mapping
    non_na = pd.Series(mapped_final, dtype="object").dropna()
    dup_vals = non_na[non_na.duplicated(keep=False)]
    if len(dup_vals) > 0:
        dup_symbols = dup_vals.unique().tolist()
        msg = (
            f"{len(dup_symbols)} duplicated mapped symbols detected "
            f"(examples: {dup_symbols[:10]}). "
            "This may be a problem if ppi_graph assumes unique gene names."
        )
        if strict_duplicates:
            raise ValueError(msg)
        elif verbose:
            print("[map_ensembl_gene_list_to_symbols] WARNING:", msg)

    if verbose:
        n_total = len(genes)
        n_ens = int(ens_mask.sum())
        n_mapped = int(pd.Series(mapped_raw, dtype="object").notna().sum())
        n_unmapped = n_ens - n_mapped
        print(
            f"[map_ensembl_gene_list_to_symbols] "
            f"{n_mapped}/{n_ens} Ensembl IDs mapped; "
            f"{n_unmapped} unmapped; total genes = {n_total}."
        )

    return mapped_final, mapping_df

# ----------------------------- CONFIG ------------------------------------------
n_top_genes = 7500
w_inter = 1.0
gamma1 = 0.5
core_rank = 30
ppi_topk = 650
K_MNN = 20
spot_num = 5000
min_cell_number_in_spot = 2
max_cell_number_in_spot = 15
K_DECONV = 10
max_cell_types_in_spot = 3
RADIUS_INTRA_ST = 140
metric = 'cosine'
ALPHA = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

ppi_csv = "/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/ppi_genes.csv"

ST_PATH = "/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/4.Human_Lymph_Node/ST.h5ad"
SC_PATH = "/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/4.Human_Lymph_Node/scRNA.h5ad"

OUT_AUPRC_CSV = "/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/ST_deconv_auprc_runs2.csv"

GT_PATH = "/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/4.Human_Lymph_Node/manual_GC_annot.csv"

GC_TARGETS = [
    "B_GC_DZ",
    "B_GC_LZ",
    "B_GC_prePB",
    "T_CD4+_TfH_GC"
]
# ----------------------------- MAIN --------------------------------------------
ST_adata = sc.read_h5ad(ST_PATH)
ST_adata.var_names_make_unique()

# ORDER-BASED ground truth attach (no barcode matching)
ground_truth = pd.read_csv(GT_PATH, sep=",")
if "cell_type" not in ground_truth.columns:
    raise KeyError(f"Expected column 'cell_type' in {GT_PATH}. Columns: {list(ground_truth.columns)}")

if len(ground_truth) != ST_adata.n_obs:
    raise ValueError(
        f"GT rows ({len(ground_truth)}) != ST spots ({ST_adata.n_obs}). "
        f"You said they're same order; they must also be same length."
    )

ST_adata.obs["ground_truth"] = (
    pd.to_numeric(ground_truth["cell_type"], errors="coerce").fillna(0).astype(int).to_numpy()
)

sc_ref = sc.read_h5ad(SC_PATH)
sc_ref.var_names_make_unique()
sc_ref.obs_names_make_unique()

# DO NOT change this (per your instruction)
label_candidates = ["cell_type"]
label_key = next((k for k in label_candidates if k in sc_ref.obs.columns), None)
if label_key is None:
    raise KeyError(
        f"Could not find a cell-type label column in sc_ref.obs. "
        f"Tried {label_candidates}. Available: {list(sc_ref.obs.columns)[:30]}"
    )

sc_ref.obs["cell_types"] = sc_ref.obs[label_key].astype("category")
sc_ref.obs["cell_type_idx"] = sc_ref.obs["cell_types"].cat.codes.astype(int)
idx_to_word_celltype = dict(enumerate(sc_ref.obs["cell_types"].cat.categories))

# 1) Pseudo-spots from scRNA reference
pseudo_spots = pseudo_spot_generation(
    sc_exp=sc_ref,
    idx_to_word_celltype=idx_to_word_celltype,
    spot_num=spot_num,
    min_cell_number_in_spot=min_cell_number_in_spot,
    max_cell_number_in_spot=max_cell_number_in_spot,
    max_cell_types_in_spot=max_cell_types_in_spot,
    generation_method="celltype"
)
if pseudo_spots.n_obs == 0:
    raise RuntimeError("No pseudo spots generated.")

# 2) Align genes across ST and pseudo-spots
genes0 = ST_adata.var_names.intersection(pseudo_spots.var_names)
if len(genes0) == 0:
    raise RuntimeError("No overlapping genes between ST and pseudo spots.")

adata_st0 = ST_adata[:, genes0].copy()
pseudo_spots = pseudo_spots[:, genes0].copy()

# 3) Joint PCA embedding for MNN edges (marker genes only)
adata_joint = anndata.concat(
    {"st": adata_st0, "pseudo": pseudo_spots},
    join="inner",
    label="domain",
    index_unique=None,
)

#marker_genes, genes_dict = find_marker_genes(
 #   sc_ref,  top_gene_per_type=TOP_MARKERS_PER_TYPE
#)
#marker_genes = [g for g in marker_genes if g in adata_joint.var_names]
#if len(marker_genes) < 5:
 #   raise RuntimeError(f"Too few marker genes available for MNN embedding: {len(marker_genes)}")

#adata_joint_mnn = adata_joint[:, marker_genes].copy()
adata_tmp = adata_joint.copy()
sc.pp.normalize_total(adata_tmp, target_sum=1e4)
sc.pp.log1p(adata_tmp)
n_top = min(n_top_genes, adata_joint.n_vars)
sc.pp.highly_variable_genes(
    adata_tmp,
    n_top_genes=n_top,
    batch_key="domain",
    flavor="seurat",
    inplace=True
)
adata_joint_mnn = adata_tmp[:, adata_tmp.var["highly_variable"]].copy()

Xj = adata_joint_mnn.X
n_pcs = min(50, adata_joint_mnn.n_vars - 1)
if n_pcs < 2:
    raise RuntimeError("Too few marker genes to compute PCA/SVD for MNN embedding.")

if sp.issparse(Xj):
    reducer = TruncatedSVD(n_components=n_pcs, random_state=SEED)
    emb_joint = reducer.fit_transform(Xj)
else:
    reducer = PCA(n_components=n_pcs, random_state=SEED)
    emb_joint = reducer.fit_transform(np.asarray(Xj, dtype=np.float32))

N_expected = adata_st0.n_obs + pseudo_spots.n_obs
assert emb_joint.shape[0] == N_expected, (emb_joint.shape, N_expected)

# 4) Build L_s on [ST | pseudo]
L_s, ns_st, rows_st, row_pseudo = build_L_st_with_pseudo_mnn(
    st_adatas=[adata_st0],
    pseudo_adata=pseudo_spots,
    emb_joint=emb_joint,
    radius_intra_st=RADIUS_INTRA_ST,
    k_mnn=K_MNN,
    intra_weight="binary",
    mnn_weight="binary",
    sigma_intra=None,
    sigma_mnn=None,
    w_intra=1.0,
    w_mnn=w_inter,
    normalize="sym",
    mnn_metric=metric,
)

S_expected = adata_st0.n_obs + pseudo_spots.n_obs
assert L_s.shape == (S_expected, S_expected), (L_s.shape, S_expected)

# 5) Preprocess the TWO datasets you will factorize
adata_st_p = _preprocess_and_hvg(adata_st0.copy(), n_top_genes=n_top_genes)
pseudo_spots_p = _preprocess_and_hvg(pseudo_spots.copy(), n_top_genes=n_top_genes)

adatas_aligned, genes = intersect_hvgs_and_align([adata_st_p, pseudo_spots_p])
if len(genes) == 0:
    raise RuntimeError("No aligned HVGs after preprocessing.")

# map aligned Ensembl IDs -> symbols, preserving order
genes_for_ppi, gene_map_df = map_ensembl_gene_list_to_symbols(
    genes,
    species="human",
    batch_size=1000,
    keep_unmapped="original",   # preserves length/order
    strict_duplicates=False,    # set True if you want to fail on duplicate symbols
    verbose=True,
)

# GENE LAPLACIAN
L_g = ppi_graph(genes_for_ppi, ppi_csv, score_threshold=ppi_topk)

# TORCH SPARSE CONVERSION
Ls_coo = L_s.tocoo()
idx = torch.from_numpy(np.vstack((Ls_coo.row, Ls_coo.col)).astype(np.int64)).to(device)
val = torch.from_numpy(Ls_coo.data.astype(np.float64)).to(device)
Ls_torch = torch.sparse_coo_tensor(idx, val, Ls_coo.shape).coalesce()

Lg_coo = sp.coo_matrix(L_g)
idx_g = torch.from_numpy(np.vstack((Lg_coo.row, Lg_coo.col)).astype(np.int64)).to(device)
val_g = torch.from_numpy(Lg_coo.data.astype(np.float64)).to(device)
Lg_torch = torch.sparse_coo_tensor(idx_g, val_g, Lg_coo.shape).coalesce()

# 6) Build X_list (order must match L_s: [st then pseudo])
X_list, ns_list, row_slices = build_irregular_slices(adatas_aligned, device=device, dtype=dtype)
assert sum(ns_list) == S_expected, (sum(ns_list), S_expected)

# FIT PARAFAC2 (AO+ADMM)
Q_list, H, B, w_list, Y = fit_parafac2_graph(
    X_list,
    Lg_torch,
    Ls_torch,
    rank=core_rank,
    gamma=gamma1,
    iters=120,
    rho=1.0,
    device=device,
    dtype=dtype,
)

# Deconvolution in QHD space
attach_QHD_embeddings(adatas_aligned, Q_list, H, w_list, key="X_parafac2", store_normed=True)
adata_st = adatas_aligned[0]
adata_ps = adatas_aligned[1]

adata_st.write('/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/4.Human_Lymph_Node/ST_deconvolved.h5ad')

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import matplotlib.pyplot as plt

# -----------------------------
# 0) Your Y_hat columns
# -----------------------------
YHAT_COLS = [
    "Y_hat_B_Cycling","Y_hat_B_GC_DZ","Y_hat_B_GC_LZ","Y_hat_B_GC_prePB","Y_hat_B_IFN",
    "Y_hat_B_activated","Y_hat_B_mem","Y_hat_B_naive","Y_hat_B_plasma","Y_hat_B_preGC",
    "Y_hat_DC_CCR7+","Y_hat_DC_cDC1","Y_hat_DC_cDC2","Y_hat_DC_pDC","Y_hat_Endo","Y_hat_FDC",
    "Y_hat_ILC","Y_hat_Macrophages_M1","Y_hat_Macrophages_M2","Y_hat_Mast","Y_hat_Monocytes",
    "Y_hat_NK","Y_hat_NKT","Y_hat_T_CD4+","Y_hat_T_CD4+_TfH","Y_hat_T_CD4+_TfH_GC",
    "Y_hat_T_CD4+_naive","Y_hat_T_CD8+_CD161+","Y_hat_T_CD8+_cytotoxic","Y_hat_T_CD8+_naive",
    "Y_hat_T_TIM3+","Y_hat_T_TfR","Y_hat_T_Treg","Y_hat_VSMC"
]

# -----------------------------
# 1) Sharp + smooth helper
# -----------------------------
def sharp_smooth_obs_score(
    adata,
    col: str,
    *,
    p_lo: float = 0.0,
    p_hi: float = 99.0,
    mode: str = "keep",          # "keep" => outside percentile band -> 0 ; "winsor" => clip to band
    gamma: float = 2.0,
    out_sharp: str | None = None,
    out_smooth: str | None = None,
    smooth: bool = True,
    coord_type: str = "grid",
    n_neighs: int = 6,
    include_self: bool = True,
    G_key: str = "spatial_connectivities",
    dtype=np.float32,
):
    if col not in adata.obs.columns:
        raise KeyError(f"{col} not found in adata.obs")

    if mode not in {"keep", "winsor"}:
        raise ValueError('mode must be "keep" or "winsor"')

    x = pd.to_numeric(adata.obs[col], errors="coerce").to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    finite = np.isfinite(x)
    if finite.sum() == 0:
        q_lo, q_hi = 0.0, 0.0
    else:
        q_lo = np.percentile(x[finite], p_lo)
        q_hi = np.percentile(x[finite], p_hi)

    if mode == "keep":
        y = np.where((x >= q_lo) & (x <= q_hi), x, 0.0)
    else:
        y = np.clip(x, q_lo, q_hi)

    y = np.clip(y, 0.0, None)
    if gamma != 1.0:
        y = np.power(y, gamma)

    if out_sharp is None:
        out_sharp = f"{col}_sharp"
    adata.obs[out_sharp] = y.astype(dtype)

    if smooth:
        # build spatial neighbor graph if missing
        if G_key not in adata.obsp:
            import squidpy as sq
            sq.gr.spatial_neighbors(adata, coord_type=coord_type, n_neighs=n_neighs)

        G = adata.obsp[G_key].tocsr()
        if include_self:
            G = G + sp.eye(G.shape[0], format="csr")

        den = np.asarray(G.sum(axis=1)).ravel()
        den = np.clip(den, 1e-12, None)

        y_smooth = (G @ y) / den

        if out_smooth is None:
            out_smooth = f"{out_sharp}_smooth"
        adata.obs[out_smooth] = y_smooth.astype(dtype)

    return q_lo, q_hi

# -----------------------------
# 2) Run sharp+smooth for all Y_hat columns
# -----------------------------
def sharp_smooth_all_yhat(
    adata,
    yhat_cols=YHAT_COLS,
    *,
    p_lo=0.0,
    p_hi=99.0,
    mode="keep",
    gamma=2.0,
    smooth=True,
    n_neighs=6,
):
    present = [c for c in yhat_cols if c in adata.obs.columns]
    missing = [c for c in yhat_cols if c not in adata.obs.columns]
    if missing:
        print("Missing columns (skipping):", missing)

    cutoffs = {}
    for c in present:
        q_lo, q_hi = sharp_smooth_obs_score(
            adata, c,
            p_lo=p_lo, p_hi=p_hi, mode=mode, gamma=gamma,
            out_sharp=f"{c}_sharp",
            out_smooth=f"{c}_sharp_smooth",
            smooth=smooth, n_neighs=n_neighs
        )
        cutoffs[c] = (q_lo, q_hi)
    return cutoffs

# -----------------------------
# 3) Cluster using X_parafac2
# -----------------------------
def leiden_on_parafac2(
    adata,
    *,
    use_rep="X_parafac2",
    n_neighbors=15,
    metric="euclidean",
    resolution=0.5,
    neighbors_key="parafac2_nn",
    cluster_key="leiden_parafac2",
):
    if use_rep not in adata.obsm:
        raise KeyError(f"{use_rep} not found in adata.obsm")
    sc.pp.neighbors(
        adata,
        use_rep=use_rep,
        n_neighbors=n_neighbors,
        metric=metric,
        key_added=neighbors_key,
    )
    sc.tl.leiden(
        adata,
        resolution=resolution,
        neighbors_key=neighbors_key,
        key_added=cluster_key,
    )
    return cluster_key

# -----------------------------
# 5) Run everything
# -----------------------------
# (A) sharp + smooth all Y_hat
cutoffs = sharp_smooth_all_yhat(
    adata,
    p_lo=0, p_hi=99.9,      # percentile band (tune p_hi to 97/99 if you want less aggressive cutoff)
    mode="keep",          # outside band -> 0 (your original behavior)
    gamma=2,
    smooth=True,
    n_neighs=6
)

# (B) leiden on X_parafac2
cluster_key = leiden_on_parafac2(
    adata,
    n_neighbors=15,
    resolution=0.7,
    cluster_key="leiden_parafac2"
)

