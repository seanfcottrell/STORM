#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata
import scipy.sparse as sp

from STORM.utils import (
    _preprocess_and_hvg,
    make_synthetic_spots_from_single_cells,
)
from STORM.Utils.TensorConstructionUtils import (
    build_irregular_slices,
    intersect_hvgs_and_align,
)
from STORM.parafac2 import fit_parafac2_graph
from STORM.Utils.PseudoSpotUtils import pseudo_spot_generation
from STORM.Utils.DeconvolutionUtils import deconvolve_gbdt_from_pseudospots, deconvolve_lle_from_pseudospots, filter_rare_celltypes_cells
from STORM.GraphConstructions import ppi_graph, build_L_st_with_pseudo_mnn
from STORM.Utils.TensorDecompositionUtils import attach_QHD_embeddings
from STORM.Metrics import evaluate_deconvolution,per_type_metrics_to_df,slice_type_means
from sklearn.decomposition import PCA, TruncatedSVD

# ----------------------------- CLI ---------------------------------------------
n_top_genes = 7500
gamma1 = 0.5
core_rank = 30
ppi_topk = 650
BIN_SIZE = 0.025
K_MNN = 20
spot_num = 2000
min_cell_number_in_spot = 2
max_cell_number_in_spot = 15
K_DECONV = 10
max_cell_types_in_spot = 3
RADIUS_INTRA_ST = 0.02
metric = 'cosine'
SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

ppi_csv = "/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/ppi_genes.csv"

# ----------------------------- config ------------------------------------------
RARE_CT_FRAC = 0.03
MIN_CELLS_PER_SYN_SPOT = 2

# ----------------------------- MAIN --------------------------------------------
ST_adata = sq.datasets.merfish()
ST_adata.var_names_make_unique()

slices = ST_adata.obs["Bregma"].unique()

# Accumulate per-slice summaries and per-(slice, celltype) metrics
slice_summaries = []   # one row per slice averaged over cell types
per_type_rows = []     

for sl in slices:
    print("\n" + "=" * 80)
    print(f"Processing slice (Bregma) = {sl}")
    print("=" * 80)

    # --- slice subset
    ST_adata_slice = ST_adata[ST_adata.obs.Bregma == sl].copy()

    adata_cells = ST_adata_slice.copy()
    ct = adata_cells.obs["Cell_class"].astype(str)
    mapping = {
    "Endothelial 1": "Endothelial",
    "Endothelial 2": "Endothelial",
    "Endothelial 3": "Endothelial",
    "OD Mature 1": "OD Mature",
    "OD Mature 2": "OD Mature",
    "OD Mature 3": "OD Mature",
    "OD Mature 4": "OD Mature",
    "OD Immature 1": "OD Immature",
    "OD Immature 2": "OD Immature",
    "OD Immature 3": "OD Immature",
    "OD Immature 4": "OD Immature",
    }
    adata_cells.obs["cell_type_binned"] = ct.replace(mapping)

    # rare cell type filter
    adata_cells, kept_celltypes = filter_rare_celltypes_cells(
        adata_cells, 'cell_type_binned', min_frac=RARE_CT_FRAC
    )
    print(
        f"[rare-ct] kept {len(kept_celltypes)} cell types (>= {RARE_CT_FRAC*100:.1f}%), "
        f"cells now {adata_cells.n_obs}"
    )

    # -----------------------------
    # 1) Synthetic ST spots (~50 microns in diameter) + GT Y_true
    # -----------------------------
    adata_syn = make_synthetic_spots_from_single_cells(
        adata_cells,
        bin_size=BIN_SIZE,
        min_cells_per_spot=2,
        layer=None,
        cell_type_key='cell_type_binned',
    )

    if "n_cells" in adata_syn.obs:
        keep_spots = (adata_syn.obs["n_cells"].to_numpy() >= MIN_CELLS_PER_SYN_SPOT)
        adata_syn = adata_syn[keep_spots].copy()
    else:
        raise KeyError("Expected adata_syn.obs['n_cells'] missing; check synthetic spot generator.")

    print(f"[syn] bin_size={BIN_SIZE}  spots={adata_syn.n_obs}  genes={adata_syn.n_vars}")
    if adata_syn.n_obs == 0:
        print("[skip] no synthetic spots for this slice")
        continue

    # -----------------------------
    # 2) Pseudo-spots from (non-spatial) single-cell ref
    # -----------------------------
    sc_ref = adata_cells.copy()
    sc_ref.obs["cell_type"] = sc_ref.obs['cell_type_binned'].astype("category")
    sc_ref.obs["cell_type_idx"] = sc_ref.obs["cell_type"].cat.codes.astype(int)
    idx_to_word_celltype = dict(enumerate(sc_ref.obs["cell_type"].cat.categories))

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
        print("[skip] no pseudo spots generated for this slice")
        continue

    # -----------------------------
    # 3) Align genes across synthetic ST and pseudo-spots
    # -----------------------------
    genes0 = adata_syn.var_names.intersection(pseudo_spots.var_names)
    if len(genes0) == 0:
        print("[skip] no overlapping genes between synthetic ST and pseudo-spots")
        continue
    adata_syn = adata_syn[:, genes0].copy()
    pseudo_spots = pseudo_spots[:, genes0].copy()

    # -----------------------------
    # 4) Joint PCA embedding for MNN 
    # -----------------------------
    adata_joint = anndata.concat(
        {"st": adata_syn, "pseudo": pseudo_spots},
        join="inner",
        label="domain",
        index_unique=None,
    )

    sc.pp.normalize_total(adata_joint, target_sum=1e4)
    sc.pp.log1p(adata_joint)

    n_top = min(7500, adata_joint.n_vars)
    sc.pp.highly_variable_genes(
        adata_joint, n_top_genes=n_top, batch_key="domain", inplace=True
    )
    adata_joint = adata_joint[:, adata_joint.var["highly_variable"]].copy()

    Xj = adata_joint.X
    n_pcs = min(50, adata_joint.n_vars - 1)
    if n_pcs < 2:
        raise RuntimeError("Too few marker genes to compute PCA/SVD for MNN embedding.")

    if sp.issparse(Xj):
        reducer = TruncatedSVD(n_components=n_pcs, random_state=SEED)
        emb_joint = reducer.fit_transform(Xj)
    else:
        reducer = PCA(n_components=n_pcs, random_state=SEED)
        emb_joint = reducer.fit_transform(np.asarray(Xj, dtype=np.float32))

    # -----------------------------
    # 5) Build L_s on [adata_syn | pseudo_spots]
    # -----------------------------
    L_s, ns_st, rows_st, row_pseudo = build_L_st_with_pseudo_mnn(
        st_adatas=[adata_syn],
        pseudo_adata=pseudo_spots,
        emb_joint=emb_joint,
        radius_intra_st=RADIUS_INTRA_ST,
        k_mnn=K_MNN,
        mnn_metric=metric
    )

    # -----------------------------
    # 6) Preprocess the datasets we will factorize
    # -----------------------------
    adata_syn_p = _preprocess_and_hvg(adata_syn.copy(), n_top_genes=n_top_genes)
    pseudo_spots_p = _preprocess_and_hvg(pseudo_spots.copy(), n_top_genes=n_top_genes)

    adatas = [adata_syn_p, pseudo_spots_p]
    adatas_aligned, genes = intersect_hvgs_and_align(adatas)
    if len(genes) == 0:
        print("[skip] no aligned HVGs after preprocessing")
        continue

    # ----------------------------- GENE LAPLACIAN ------------------------------
    L_g = ppi_graph(genes, ppi_csv, score_threshold=ppi_topk)

    # ----------------------------- device
    Ls_coo = L_s.tocoo()
    idx = torch.from_numpy(np.vstack((Ls_coo.row, Ls_coo.col)).astype(np.int64)).to(device)
    val = torch.from_numpy(Ls_coo.data.astype(np.float64)).to(device)
    Ls_torch = torch.sparse_coo_tensor(idx, val, Ls_coo.shape).coalesce()

    Lg_coo = sp.coo_matrix(L_g)
    idx_g = torch.from_numpy(np.vstack((Lg_coo.row, Lg_coo.col)).astype(np.int64)).to(device)
    val_g = torch.from_numpy(Lg_coo.data.astype(np.float64)).to(device)
    Lg_torch = torch.sparse_coo_tensor(idx_g, val_g, Lg_coo.shape).coalesce()

    # -----------------------------
    # 7) Build X_list (order must match L_s: [st then pseudo])
    # -----------------------------
    X_list, ns_list, row_slices = build_irregular_slices(
        adatas_aligned, device=device, dtype=dtype
    )

    # ----------------------------- FIT STORM Model ----------------------
    Q_list, H, B, w_list, Y = fit_parafac2_graph(
        X_list,
        Lg_torch,
        Ls_torch,
        rank=core_rank,
        gamma=gamma1,
        iters=30,
        rho=1.0,
        device=device,
        dtype=dtype,
    )

    # ----------------------------- Deconvolution in QHD space ------------------
    attach_QHD_embeddings(
        adatas_aligned,
        Q_list,
        H,
        w_list,
        key="X_parafac2",
        also_store_shape=False,
        store_normed=True,
    )

    adata_st = adatas_aligned[0]  # synthetic ST (HVG-aligned)
    adata_ps = adatas_aligned[1]  # pseudo (HVG-aligned)

    # propagate Y_true from original adata_syn to adata_st
    if "Y_true" in adata_syn.obsm:
        adata_st.obsm["Y_true"] = np.asarray(adata_syn.obsm["Y_true"], dtype=np.float32)
        adata_st.uns["Y_true_celltypes"] = list(adata_syn.uns["Y_true_celltypes"])
    else:
        raise KeyError(
            "Ground truth Y_true missing from adata_syn.obsm. "
            "Check cell_type_key passed to make_synthetic_spots_from_single_cells."
        )
    celltype_cols = [idx_to_word_celltype[i] for i in range(len(idx_to_word_celltype))]
    celltype_cols = [c for c in celltype_cols if c in adata_ps.obs.columns]
    # predict
    # --- LLE ---
    Y_lle = deconvolve_lle_from_pseudospots(
        adata_st,
        adata_ps,
        emb_key="X_parafac2_rownorm",
        k=K_DECONV,          # reuse your K_DECONV
        reg=1e-3,
        out_key="Y_hat_lle",
        celltype_cols=celltype_cols
    )

    # --- GBDT ---
    #Y_gbdt = deconvolve_gbdt_from_pseudospots(
     #   adata_st,
      #  adata_ps,
       # emb_key="X_parafac2_rownorm",
        #out_key="Y_hat_gbdt",
        #celltype_cols=celltype_cols,
    #)

    adata_st.obsm["Y_hat"] = Y_lle # for synthetic data we use only kNN regression
    adata_st.uns["Y_hat_celltypes"] = list(celltype_cols)
    # evaluate
    metrics = evaluate_deconvolution(adata_st, true_key="Y_true", pred_key="Y_hat")

    print(
        f"[slice={sl}] RMSE={metrics['rmse_spot_mean']:.4f}  "
        f"JSD={metrics['jsd_spot_mean']:.4f}  "
    )
