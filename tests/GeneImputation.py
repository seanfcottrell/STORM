#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import gc
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scipy.sparse as sp

from STORM.utils import _preprocess_and_hvg
from STORM.Utils.TensorConstructionUtils import (
    build_irregular_slices,
    intersect_hvgs_and_align,
)
from STORM.Utils.TensorDecompositionUtils import attach_QHD_embeddings
from STORM.GraphConstructions import ppi_graph, build_L_st_with_pseudo_mnn, _intra_adj_from_coords
from STORM.parafac2 import fit_parafac2_graph
from STORM.Utils.PseudoSpotUtils import pseudo_spot_generation
from sklearn.decomposition import PCA, TruncatedSVD
from STORM.Utils.ImputationUtils import (
    knn_pseudo_indices_cosine_torch,
    build_X_knn_from_pseudo_expr,
    cal_B_all_and_gene_graph,
    scipy_sparse_to_torch_coo
)
from STORM.ImputationGNN import (
    fit_gene_imputer,
    predict_gene_imputer,
)

# ----------------------------- CLI / CONFIG -----------------------------------
n_top_genes = 7500
gamma1 = 0.5
core_rank = 30
ppi_topk = 650

alpha = 0.9
K_MNN = 20
spot_num = 8000
GENE_GRAPH_k = 20
B_RIDGE_LAM = 1
GNN_HIDDEN = 128
max_cell_types_in_spot = 3
'''spatial radii:
1.  merfish       13.0
2.  osmFISH Allen SSP      170.0
3.  osmFISH Allen VIsp      190.0
4.  osmFISH Zeisel      190.0
5.  starmap      140.0
'''
RADIUS_INTRA_ST = 190
metric = "cosine"
# GNN / graph hyperparams
K_GNN = 50
GNN_EPOCHS = 50
GNN_BATCH = 64
GNN_LR = 1e-3
GNN_WD = 1e-2
GNN_LAYERS = 1
GNN_SIM_WEIGHT = 0.5

SEED = 0
N_FOLDS = 5
TARGET_SUM_DEFAULT = 1e4  # fallback 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

ppi_csv = "/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/ppi_genes.csv"
# cell type keys in single cell references 
CT_KEY = "level1class" #zeisel
#CT_KEY = 'clusters' #moffit
#CT_KEY = "subclass" #visp
#CT_KEY = 'subclass_label' #ssp

# ----------------------------- evaluation utilities --------------------------------------
def _to_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)

def _median_target_sum_from_obs(counts_obs, fallback: float = TARGET_SUM_DEFAULT) -> float:
    """
    normalization uses median(total_counts_per_cell) as target_sum
    """
    if sp.issparse(counts_obs):
        denom = np.asarray(counts_obs.sum(axis=1)).reshape(-1).astype(np.float64)
    else:
        denom = np.asarray(counts_obs, dtype=np.float64).sum(axis=1)

    denom_pos = denom[denom > 0]
    if denom_pos.size == 0:
        return float(fallback)
    return float(np.median(denom_pos))

def _normlog_with_obs_sf(counts_obs, counts_hold, target_sum: float):
    """
    Normalize using observed genes (counts_obs), apply same size factor
    to held-out genes (counts_hold), then log1p.

      sf_i = target_sum / sum_j counts_obs[i,j]
      return log1p(counts_hold * sf_i)
    """
    counts_obs = _to_dense(counts_obs).astype(np.float64)
    counts_hold = _to_dense(counts_hold).astype(np.float64)

    denom = counts_obs.sum(axis=1)
    sf = target_sum / np.clip(denom, 1e-8, None)
    return np.log1p(counts_hold * sf[:, None]).astype(np.float32)

def calc_all_np(mat_true: np.ndarray, mat_pred: np.ndarray, cal: str = "cosine", eps: float = 1e-12):
    """
    compute metric column-wise.

    If cal == 'mse':
        returns column-wise MSE (length = n_cols)

    If cal == 'cosine':
        returns column-wise cosine similarity = 1 - cosine_distance
        (length = n_cols)
    """
    A = mat_true.astype(np.float64, copy=False)
    B = mat_pred.astype(np.float64, copy=False)
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    if cal == "mse":
        return np.mean((A - B) ** 2, axis=0)

    if cal == "cosine":
        num = np.sum(A * B, axis=0)
        den = (np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0))
        sim = num / (den + eps)
        sim = np.where(den > 0, sim, 0.0)
        return sim

    raise ValueError("cal must be one of {'cosine','mse'}")

def metrics(Y_true: np.ndarray, Y_pred: np.ndarray):
    """
      gene-wise cosine: median over genes of cosine(true[:,g], pred[:,g])
      gene-wise mse:    median over genes of mse(true[:,g], pred[:,g])
      cell-wise cosine: median over cells of cosine(true[c,:], pred[c,:])
      cell-wise mse:    median over cells of mse(true[c,:], pred[c,:])
    """
    gene_cos = np.median(calc_all_np(Y_true, Y_pred, cal="cosine"))
    gene_mse = np.median(calc_all_np(Y_true, Y_pred, cal="mse"))
    cell_cos = np.median(calc_all_np(Y_true.T, Y_pred.T, cal="cosine"))
    cell_mse = np.median(calc_all_np(Y_true.T, Y_pred.T, cal="mse"))
    return {
        "gene_wise_cosine_median": float(gene_cos),
        "gene_wise_mse_median": float(gene_mse),
        "cell_wise_cosine_median": float(cell_cos),
        "cell_wise_mse_median": float(cell_mse),
    }

def _mix_with_knn_mean(Y_hat_all: np.ndarray, X_knn: torch.Tensor, alpha):
    """
        knn_mean = mean_k neighbors in expression space
        alpha fitted on SHARED genes only
        Y_final = alpha*Y_hat + (1-alpha)*knn_mean
    """
    with torch.no_grad():
        knn_mean = X_knn.mean(dim=1).detach().cpu().numpy()  # (n_st, G_all)

    Y_final = alpha * Y_hat_all + (1.0 - alpha) * knn_mean
    return Y_final


# -----------------------------
# 0) Load ST + scRNA reference 
# -----------------------------
ST_raw = sc.read(
    "/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/SpaGE Datasets/Spatial/osmFISH/adata.h5ad"
)
ST_raw.var_names_make_unique()
adata_cells = sc.read(
    "/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/SpaGE Datasets/scRNAseq/Zeisel/adata.h5ad"
)
adata_cells.var_names_make_unique()

# sanitize gene strings early
ST_raw.var_names = ST_raw.var_names.astype(str).str.strip()
adata_cells.var_names = adata_cells.var_names.astype(str).str.strip()

# full ST truth holder
ST_full = ST_raw.copy()
ST_full.var_names = ST_full.var_names.astype(str).str.strip()

# -----------------------------
# 1) Build shared gene set + 5-fold partition 
# -----------------------------
rng = np.random.default_rng(SEED)
shared = ST_raw.var_names.intersection(adata_cells.var_names)
if len(shared) < 10:
    raise ValueError(f"Too few shared genes for 5-fold benchmarking: {len(shared)}")

shared_list = list(shared)
perm = rng.permutation(len(shared_list))
fold_indices = np.array_split(perm, N_FOLDS)

print(f"[benchmark] shared_genes={len(shared_list)} folds={N_FOLDS} seed={SEED}")

# -----------------------------
# 2) Pseudo-spots from scRNA reference
# -----------------------------
sc_ref = adata_cells.copy()
sc_ref.obs[CT_KEY] = sc_ref.obs[CT_KEY].astype("category") 
sc_ref.obs["cell_type"] = sc_ref.obs[CT_KEY].astype("category")
sc_ref.obs["cell_type_idx"] = sc_ref.obs[CT_KEY].cat.codes.astype(int)
idx_to_word_celltype = dict(enumerate(sc_ref.obs[CT_KEY].cat.categories))
pseudo_spots0 = pseudo_spot_generation(
    sc_exp=sc_ref,
    idx_to_word_celltype=idx_to_word_celltype,
    spot_num=spot_num,
    min_cell_number_in_spot=2,
    max_cell_number_in_spot=15,
    max_cell_types_in_spot=max_cell_types_in_spot,
    generation_method="celltype",
)

pseudo_full = pseudo_spots0.copy()
pseudo_full.var_names = pseudo_full.var_names.astype(str).str.strip()

# ----------------------------- fold runner ------------------------------------
def run_one_fold(fold_id: int, holdout_genes: pd.Index):
    """
    One fold (with cell-filtering fix + improvements 1 & 2):
      - mask holdout genes from ST
      - run _preprocess_and_hvg to determine which obs survive filtering
      - subset to surviving obs
      - (1) compute per-fold target_sum as median library size on observed genes (ST, genes0)
      - build L_s on surviving obs
      - train PARAFAC2 + B-extension + gene-graph + GNN
      - (2) final mixing: alpha*model_pred + (1-alpha)*knn_mean (alpha fit on shared genes)
      - compute metrics on heldout genes
    """
    print(f"\n==================== FOLD {fold_id+1}/{N_FOLDS} ====================")
    print(f"[fold] holdout genes = {len(holdout_genes)}")

    # per-fold determinism
    np.random.seed(SEED + fold_id)
    torch.manual_seed(SEED + fold_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + fold_id)

    # 1) Mask ST: drop heldout genes
    measured_genes = pd.Index([g for g in ST_raw.var_names if g not in set(holdout_genes)])
    ST_mask0 = ST_raw[:, measured_genes].copy()
    ST_mask0.var_names = ST_mask0.var_names.astype(str).str.strip()

    # genes0 = measured shared genes 
    genes0 = ST_mask0.var_names.intersection(pseudo_full.var_names)
    if len(genes0) == 0:
        raise RuntimeError("No overlapping measured genes between masked ST and pseudo spots.")
    print(f"[fold] measured shared genes (genes0) = {len(genes0)}")

    pseudo_measured0 = pseudo_full[:, genes0].copy()

    # 2) Determine which obs survive filtering
    st_tmp = _preprocess_and_hvg(ST_mask0.copy(), n_top_genes=n_top_genes)
    ps_tmp = _preprocess_and_hvg(pseudo_measured0.copy(), n_top_genes=n_top_genes)

    kept_st = st_tmp.obs_names
    kept_ps = ps_tmp.obs_names

    if len(kept_st) != ST_mask0.n_obs or len(kept_ps) != pseudo_measured0.n_obs:
        print(f"[fold] filtered obs: ST {ST_mask0.n_obs}->{len(kept_st)} | pseudo {pseudo_measured0.n_obs}->{len(kept_ps)}")

    # 3) Subset ALL relevant objects to kept obs
    ST_mask = ST_mask0[kept_st, :].copy()          # measured genes only
    ST_truth = ST_full[kept_st, :].copy()          # full truth, same cells as ST_mask
    pseudo_full_k = pseudo_full[kept_ps, :].copy() # full pseudo, same spots as embeddings
    pseudo_measured = pseudo_full_k[:, genes0].copy()

    # (1) per-fold target_sum = median library size on observed genes (ST genes0)
    target_sum_fold = _median_target_sum_from_obs(ST_mask[:, genes0].X, fallback=TARGET_SUM_DEFAULT)

    # 4) Build the TWO preprocessed objects we will factorize (on kept obs)
    adata_syn_p = _preprocess_and_hvg(ST_mask.copy(), n_top_genes=n_top_genes)
    pseudo_spots_p = _preprocess_and_hvg(pseudo_measured.copy(), n_top_genes=n_top_genes)

    # 5) Joint PCA embedding for MNN edges 
    adata_st0 = ST_mask[:, genes0].copy()
    adata_joint = anndata.concat(
        {"st": adata_st0, "pseudo": pseudo_measured},
        join="inner",
        label="domain",
        index_unique=None,
    )

    sc.pp.normalize_total(adata_joint, target_sum=target_sum_fold)
    sc.pp.log1p(adata_joint)

    n_top = min(n_top_genes, adata_joint.n_vars)
    sc.pp.highly_variable_genes(
        adata_joint,
        n_top_genes=n_top,
        batch_key="domain",
        inplace=True,
    )
    adata_joint = adata_joint[:, adata_joint.var["highly_variable"]].copy()

    X = adata_joint.X
    n_pcs = min(20, adata_joint.n_vars - 1)
    if n_pcs < 2:
        raise RuntimeError("Too few genes to compute PCA/SVD for MNN embedding.")

    if sp.issparse(X):
        reducer = TruncatedSVD(n_components=n_pcs, random_state=42)
        adata_joint.obsm["X_pca"] = reducer.fit_transform(X)
    else:
        reducer = PCA(n_components=n_pcs, random_state=42)
        adata_joint.obsm["X_pca"] = reducer.fit_transform(np.asarray(X, dtype=np.float32))

    emb_joint = adata_joint.obsm["X_pca"]  # row order = [st then pseudo]

    # 6) Build L_s on [ST | pseudo] (kept obs)
    L_s, _, _, _ = build_L_st_with_pseudo_mnn(
        st_adatas=[adata_st0],
        pseudo_adata=pseudo_measured,
        emb_joint=emb_joint,
        radius_intra_st=RADIUS_INTRA_ST,
        k_mnn=K_MNN,
        mnn_metric=metric
    )

    S_expected = adata_st0.n_obs + pseudo_measured.n_obs
    assert L_s.shape == (S_expected, S_expected), (L_s.shape, S_expected)

    # 7) Align HVGs/genes for factorization
    adatas_aligned, genes = intersect_hvgs_and_align([adata_syn_p, pseudo_spots_p])
    if len(genes) == 0:
        raise RuntimeError("No aligned HVGs after preprocessing.")
    g_s = len(genes)

    # Ensure row order used by factorization matches L_s row order
    if list(adatas_aligned[0].obs_names) != list(adata_st0.obs_names):
        raise RuntimeError("Row-order mismatch: adatas_aligned[0] != adata_st0 (will break L_s alignment).")
    if list(adatas_aligned[1].obs_names) != list(pseudo_measured.obs_names):
        raise RuntimeError("Row-order mismatch: adatas_aligned[1] != pseudo_measured (will break L_s alignment).")

    # 8) Gene Laplacian (PPI)
    L_g = ppi_graph(genes, ppi_csv, score_threshold=ppi_topk)

    # 9) device
    Ls_coo = L_s.tocoo()
    idx = torch.from_numpy(np.vstack((Ls_coo.row, Ls_coo.col)).astype(np.int64)).to(device)
    val = torch.from_numpy(Ls_coo.data.astype(np.float64)).to(device)
    Ls_torch = torch.sparse_coo_tensor(idx, val, Ls_coo.shape).coalesce()

    Lg_coo = sp.coo_matrix(L_g)
    idx_g = torch.from_numpy(np.vstack((Lg_coo.row, Lg_coo.col)).astype(np.int64)).to(device)
    val_g = torch.from_numpy(Lg_coo.data.astype(np.float64)).to(device)
    Lg_torch = torch.sparse_coo_tensor(idx_g, val_g, Lg_coo.shape).coalesce()

    # 10) Build X_list (order must match L_s: [st then pseudo])
    X_list, ns_list, _ = build_irregular_slices(adatas_aligned, device=device, dtype=dtype)
    assert sum(ns_list) == S_expected, (sum(ns_list), S_expected)

    # 11) Fit STORM
    Q_list, H, B, w_list, _ = fit_parafac2_graph(
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

    # 12) Attach QHD embeddings
    attach_QHD_embeddings(
        adatas_aligned,
        Q_list,
        H,
        w_list,
        key="X_parafac2",
        also_store_shape=False,
        store_normed=True,
    )

    adata_st = adatas_aligned[0]
    adata_ps = adatas_aligned[1]
    C_st = np.asarray(adata_st.obsm["X_parafac2_rownorm"], dtype=np.float32)
    C_ps = np.asarray(adata_ps.obsm["X_parafac2_rownorm"], dtype=np.float32)

    coords_st = adata_st.obsm['spatial']  # (n_st, 2) in same order as adata_st.obs_names
    A_spatial = _intra_adj_from_coords(
        coords_st,
        radius=RADIUS_INTRA_ST
    )  # scipy CSR (n_st, n_st)

    spot_graph = scipy_sparse_to_torch_coo(A_spatial, device=device, dtype=torch.float32)
    # 13) Holdout genes
    hold_eval = holdout_genes.intersection(ST_truth.var_names).intersection(pseudo_full_k.var_names)
    if len(hold_eval) == 0:
        raise RuntimeError("hold_eval is empty.")
    imp_genes = list(hold_eval)
    g_u = len(imp_genes)

    # 14) Build normalized matrices 
    ps_obs = pseudo_full_k[:, genes0].X
    ps_hold = pseudo_full_k[:, imp_genes].X
    Y_ps_unshared = _normlog_with_obs_sf(ps_obs, ps_hold, target_sum=target_sum_fold)  # (n_pseudo x g_u)

    st_obs = ST_mask[:, genes0].X
    st_hold = ST_truth[:, imp_genes].X
    Y_st_true = _normlog_with_obs_sf(st_obs, st_hold, target_sum=target_sum_fold)      # (n_st x g_u)

    st_shared = _normlog_with_obs_sf(st_obs, ST_mask[:, genes].X, target_sum=target_sum_fold)          # (n_st x g_s)
    ps_shared = _normlog_with_obs_sf(ps_obs, pseudo_full_k[:, genes].X, target_sum=target_sum_fold)    # (n_pseudo x g_s)

    # 15) Infer B_unshared + build gene kNN graph
    X_pseudo_unshared_torch = torch.as_tensor(Y_ps_unshared, device=device, dtype=H.dtype)

    B_all, gene_graph = cal_B_all_and_gene_graph(
        Q_list=Q_list,
        H=H,
        w_list=w_list,
        B_shared=B,  # (g_s x R), rows correspond to `genes`
        X_pseudo_unshared=X_pseudo_unshared_torch,
        pseudo_index=1,
        lam=B_RIDGE_LAM,
        knn_k=int(GENE_GRAPH_k),
        chunk_size=2048,
    )

    G_all = g_s + g_u

    # 16) Build kNN pseudo-neighbor tensor X_knn: (n_st, k, G_all)
    idx_st_to_pseudo = knn_pseudo_indices_cosine_torch(
        C_st=C_st,
        C_ps=C_ps,
        k=K_GNN,
        device=device,
        batch=1024,
    )

    pseudo_expr_all = np.concatenate([ps_shared, Y_ps_unshared], axis=1).astype(np.float32)

    X_knn = build_X_knn_from_pseudo_expr(
        pseudo_expr_all=pseudo_expr_all,
        idx_st_to_pseudo=idx_st_to_pseudo,
    )

    # 17) Train gene-graph GNN
    Y_shared_torch = torch.as_tensor(st_shared, device=device, dtype=torch.float32)
    shared_idx = torch.arange(g_s, device=device, dtype=torch.long)
    gene_graph = gene_graph.to(device).float()

    model, adj_norm = fit_gene_imputer(
        X_knn=X_knn,
        Y_shared=Y_shared_torch,
        graph=gene_graph,
        shared_idx=shared_idx,
        spatial_graph=spot_graph,   
        epochs=GNN_EPOCHS,
        batch_size=GNN_BATCH,      
        lr=GNN_LR,
        weight_decay=GNN_WD,
        gnn_layers=GNN_LAYERS,
        mlp_hidden=GNN_HIDDEN,
        sim_weight=GNN_SIM_WEIGHT,
        device=device,
    )

    # 18) Predict all genes
    Y_hat_raw = predict_gene_imputer(model, X_knn, adj_norm, batch_size=128).detach().cpu().numpy()  # (n_st x G_all)

    # (2) mixing with kNN mean
    Y_hat_final = _mix_with_knn_mean(
        Y_hat_all=Y_hat_raw,
        X_knn=X_knn,
        alpha = alpha 
    )

    # heldout block
    Y_st_pred_holdout = Y_hat_final[:, g_s:]  # (n_st x g_u)

    # 19) metrics
    m = metrics(Y_st_true, Y_st_pred_holdout)

    print(f"gene-wise cosine: {m['gene_wise_cosine_median']:.4f}")
    print(f"gene-wise mse:    {m['gene_wise_mse_median']:.6f}")
    print(f"cell-wise cosine: {m['cell_wise_cosine_median']:.4f}")
    print(f"cell-wise mse:    {m['cell_wise_mse_median']:.6f}")

    # cleanup
    del X_list, Ls_torch, Lg_torch, gene_graph, X_knn, model, adj_norm, spot_graph
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "fold": int(fold_id),
        "n_st_used": int(ST_mask.n_obs),
        "n_pseudo_used": int(pseudo_measured.n_obs),
        "n_holdout": int(g_u),
        "target_sum_fold": float(target_sum_fold),
        "alpha_knn_mix": float(alpha),
        **m,
    }


# ----------------------------- run CV ------------------------------------------
rows = []
for f, idxs in enumerate(fold_indices):
    holdout = pd.Index([shared_list[i] for i in idxs])
    rows.append(run_one_fold(f, holdout))

df = pd.DataFrame(rows)

print("\n==================== CV SUMMARY ====================")
print(df)

summary = {
    "fold": -1,
    "n_st_used": int(df["n_st_used"].median()),
    "n_pseudo_used": int(df["n_pseudo_used"].median()),
    "n_holdout": int(df["n_holdout"].sum()),
    "target_sum_fold": float(df["target_sum_fold"].median()),
    "alpha_knn_mix": float(df["alpha_knn_mix"].mean()),
    "gene_wise_cosine_median": float(df["gene_wise_cosine_median"].mean()),
    "gene_wise_mse_median": float(df["gene_wise_mse_median"].mean()),
    "cell_wise_cosine_median": float(df["cell_wise_cosine_median"].mean()),
    "cell_wise_mse_median": float(df["cell_wise_mse_median"].mean()),
}
df2 = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

mean_including_minus1 = {
    "gene_wise_cosine_median": float(df2["gene_wise_cosine_median"].mean()),
    "gene_wise_mse_median": float(df2["gene_wise_mse_median"].mean()),
    "cell_wise_cosine_median": float(df2["cell_wise_cosine_median"].mean()),
    "cell_wise_mse_median": float(df2["cell_wise_mse_median"].mean()),
}

print("\n=== Mean across folds ===")
print(f"gene-wise cosine (mean over folds) = {summary['gene_wise_cosine_median']:.4f}")
print(f"gene-wise mse    (mean over folds) = {summary['gene_wise_mse_median']:.6f}")
print(f"cell-wise cosine (mean over folds) = {summary['cell_wise_cosine_median']:.4f}")
print(f"cell-wise mse    (mean over folds) = {summary['cell_wise_mse_median']:.6f}")

print("\n=== Mean over folds + (-1 row) ===")
print(f"gene-wise cosine (mean) = {mean_including_minus1['gene_wise_cosine_median']:.4f}")
print(f"gene-wise mse    (mean) = {mean_including_minus1['gene_wise_mse_median']:.6f}")
print(f"cell-wise cosine (mean) = {mean_including_minus1['cell_wise_cosine_median']:.4f}")
print(f"cell-wise mse    (mean) = {mean_including_minus1['cell_wise_mse_median']:.6f}")
