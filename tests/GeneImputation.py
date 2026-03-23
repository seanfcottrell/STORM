#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

import gc
import time
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scipy.sparse as sp

from STORM.STORM import STORM
from STORM.utils import _preprocess_and_hvg
from STORM.Utils.ImputationUtils import (
    knn_pseudo_indices_cosine_torch,
    build_X_knn_from_pseudo_expr,
    cal_B_all_and_gene_graph,
    scipy_sparse_to_torch_coo,
)
from STORM.GraphConstructions import _intra_adj_from_coords
from sklearn.decomposition import PCA, TruncatedSVD

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
RADIUS_INTRA_ST = 190 #osmFISH Zeisel
#RADIUS_INTRA_ST = 13 #merfish
#RADIUS_INTRA_ST = 170 #osmFISH Allen SSP
#RADIUS_INTRA_ST = 190 #osmFISH Allen VIsp
#RADIUS_INTRA_ST = 140.0 #starmap
metric = "cosine"
K_GNN = 50
GNN_EPOCHS = 50
GNN_BATCH = 64
GNN_LR = 1e-3
GNN_WD = 1e-2
GNN_LAYERS = 1
GNN_SIM_WEIGHT = 0.5

SEED = 0
N_FOLDS = 5
TARGET_SUM_DEFAULT = 1e4

device = "cpu"
dtype = torch.float64

ppi_csv = "/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/ppi_genes.csv"
CT_KEY = "level1class"

# ----------------------------- HELPERS ----------------------------------------
def _to_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)

def _median_target_sum_from_obs(counts_obs, fallback: float = TARGET_SUM_DEFAULT) -> float:
    if sp.issparse(counts_obs):
        denom = np.asarray(counts_obs.sum(axis=1)).reshape(-1).astype(np.float64)
    else:
        denom = np.asarray(counts_obs, dtype=np.float64).sum(axis=1)
    denom_pos = denom[denom > 0]
    if denom_pos.size == 0:
        return float(fallback)
    return float(np.median(denom_pos))

def _normlog_with_obs_sf(counts_obs, counts_hold, target_sum: float):
    counts_obs  = _to_dense(counts_obs).astype(np.float64)
    counts_hold = _to_dense(counts_hold).astype(np.float64)
    denom = counts_obs.sum(axis=1)
    sf = target_sum / np.clip(denom, 1e-8, None)
    return np.log1p(counts_hold * sf[:, None]).astype(np.float32)

def calc_all_np(mat_true: np.ndarray, mat_pred: np.ndarray, cal: str = "cosine", eps: float = 1e-12):
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
    gene_cos = np.median(calc_all_np(Y_true,   Y_pred,   cal="cosine"))
    gene_mse = np.median(calc_all_np(Y_true,   Y_pred,   cal="mse"))
    cell_cos = np.median(calc_all_np(Y_true.T, Y_pred.T, cal="cosine"))
    cell_mse = np.median(calc_all_np(Y_true.T, Y_pred.T, cal="mse"))
    return {
        "gene_wise_cosine_median": float(gene_cos),
        "gene_wise_mse_median":    float(gene_mse),
        "cell_wise_cosine_median": float(cell_cos),
        "cell_wise_mse_median":    float(cell_mse),
    }

def _mix_with_knn_mean(Y_hat_all: np.ndarray, X_knn: torch.Tensor, alpha):
    with torch.no_grad():
        knn_mean = X_knn.mean(dim=1).detach().cpu().numpy()
    Y_final = alpha * Y_hat_all + (1.0 - alpha) * knn_mean
    return Y_final

# ----------------------------- DATA LOADING ------------------------------------
ST_raw = sc.read(
    "/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/SpaGE Datasets/Spatial/osmFISH/adata.h5ad"
)
ST_raw.var_names_make_unique()
adata_cells = sc.read(
    "/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/SpaGE Datasets/scRNAseq/Zeisel/adata.h5ad"
)
adata_cells.var_names_make_unique()

ST_raw.var_names     = ST_raw.var_names.astype(str).str.strip()
adata_cells.var_names = adata_cells.var_names.astype(str).str.strip()

ST_full = ST_raw.copy()
ST_full.var_names = ST_full.var_names.astype(str).str.strip()

# ----------------------------- FOLD SETUP --------------------------------------
rng = np.random.default_rng(SEED)
shared = ST_raw.var_names.intersection(adata_cells.var_names)
if len(shared) < 10:
    raise ValueError(f"Too few shared genes for 5-fold benchmarking: {len(shared)}")

shared_list  = list(shared)
perm         = rng.permutation(len(shared_list))
fold_indices = np.array_split(perm, N_FOLDS)

print(f"[benchmark] shared_genes={len(shared_list)} folds={N_FOLDS} seed={SEED}")

# ----------------------------- PSEUDO-SPOTS ----------------------------
sc_ref = adata_cells.copy()
sc_ref.obs[CT_KEY]          = sc_ref.obs[CT_KEY].astype("category")
sc_ref.obs["cell_type"]     = sc_ref.obs[CT_KEY].astype("category")
sc_ref.obs["cell_type_idx"] = sc_ref.obs[CT_KEY].cat.codes.astype(int)
idx_to_word_celltype        = dict(enumerate(sc_ref.obs[CT_KEY].cat.categories))

storm_global = STORM(device=device, dtype=dtype, seed=SEED)
storm_global.sc_ref               = sc_ref
storm_global.idx_to_word_celltype = idx_to_word_celltype
storm_global.generate_pseudo_spots(
    spot_num=spot_num,
    min_cells=2,
    max_cells=15,
    max_cell_types=max_cell_types_in_spot,
    method="celltype",
)
pseudo_full = storm_global.pseudo_spots.copy()
pseudo_full.var_names = pseudo_full.var_names.astype(str).str.strip()

# ----------------------------- FOLD RUNNER ------------------------------------
def run_one_fold(fold_id: int, holdout_genes: pd.Index):
    print(f"\n==================== FOLD {fold_id+1}/{N_FOLDS} ====================")
    print(f"[fold] holdout genes = {len(holdout_genes)}")

    np.random.seed(SEED + fold_id)
    torch.manual_seed(SEED + fold_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + fold_id)

    #  Mask ST: drop holdout genes
    measured_genes = pd.Index([g for g in ST_raw.var_names if g not in set(holdout_genes)])
    ST_mask0 = ST_raw[:, measured_genes].copy()
    ST_mask0.var_names = ST_mask0.var_names.astype(str).str.strip()

    genes0 = ST_mask0.var_names.intersection(pseudo_full.var_names)

    pseudo_measured0 = pseudo_full[:, genes0].copy()

    # Determine which obs survive filtering
    st_tmp = _preprocess_and_hvg(ST_mask0.copy(),         n_top_genes=n_top_genes)
    ps_tmp = _preprocess_and_hvg(pseudo_measured0.copy(), n_top_genes=n_top_genes)

    kept_st = st_tmp.obs_names
    kept_ps = ps_tmp.obs_names

    # Subset to kept obs
    ST_mask         = ST_mask0[kept_st, :].copy()
    ST_truth        = ST_full[kept_st, :].copy()
    pseudo_full_k   = pseudo_full[kept_ps, :].copy()
    pseudo_measured = pseudo_full_k[:, genes0].copy()

    # Per-fold target_sum = median library size on observed genes
    target_sum_fold = _median_target_sum_from_obs(ST_mask[:, genes0].X, fallback=TARGET_SUM_DEFAULT)

    # Joint PCA embedding for MNN edges 
    adata_st0   = ST_mask[:, genes0].copy()
    adata_joint = anndata.concat(
        {"st": adata_st0, "pseudo": pseudo_measured},
        join="inner", label="domain", index_unique=None,
    )
    sc.pp.normalize_total(adata_joint, target_sum=target_sum_fold)
    sc.pp.log1p(adata_joint)
    n_top = min(n_top_genes, adata_joint.n_vars)
    sc.pp.highly_variable_genes(
        adata_joint, n_top_genes=n_top, batch_key="domain", inplace=True,
    )
    adata_joint = adata_joint[:, adata_joint.var["highly_variable"]].copy()

    X     = adata_joint.X
    n_pcs = min(20, adata_joint.n_vars - 1)
    if sp.issparse(X):
        reducer   = TruncatedSVD(n_components=n_pcs, random_state=42)
        emb_joint = reducer.fit_transform(X)
    else:
        reducer   = PCA(n_components=n_pcs, random_state=42)
        emb_joint = reducer.fit_transform(np.asarray(X, dtype=np.float32))

    # spatial graph + factorization + embeddings
    storm = STORM(device=device, dtype=dtype, seed=SEED + fold_id)

    storm._adata_st_genes0 = adata_st0
    storm._pseudo_genes0   = pseudo_measured
    storm._emb_joint       = emb_joint

    storm.build_spatial_graph_singlecell_reference_integration(
        radius_intra_st=RADIUS_INTRA_ST, k_mnn=K_MNN, mnn_metric=metric
    )
    storm.preprocess_and_align_singlecell_reference_integration(n_top_genes=n_top_genes)
    genes = storm.genes
    g_s   = len(genes)

    storm.build_slices_singlecell_reference_integration()
    storm.build_gene_graph(ppi_csv, score_threshold=ppi_topk)
    storm.fit(rank=core_rank, gamma=gamma1, iters=30)
    storm.attach_embeddings(key="X_parafac2")

    adata_st = storm.adatas_aligned[0]
    adata_ps = storm.adatas_aligned[1]
    C_st = np.asarray(adata_st.obsm["X_parafac2_rownorm"], dtype=np.float32)
    C_ps = np.asarray(adata_ps.obsm["X_parafac2_rownorm"], dtype=np.float32)

    coords_st = adata_st.obsm['spatial']
    A_spatial = _intra_adj_from_coords(coords_st, radius=RADIUS_INTRA_ST)
    spot_graph = scipy_sparse_to_torch_coo(A_spatial, device=device, dtype=torch.float32)

    # Holdout genes 
    hold_eval = holdout_genes.intersection(ST_truth.var_names).intersection(pseudo_full_k.var_names)
    imp_genes = list(hold_eval)
    g_u = len(imp_genes)

    ps_obs = pseudo_full_k[:, genes0].X
    ps_hold = pseudo_full_k[:, imp_genes].X
    Y_ps_unshared = _normlog_with_obs_sf(ps_obs, ps_hold, target_sum=target_sum_fold)

    st_obs = ST_mask[:, genes0].X
    st_hold = ST_truth[:, imp_genes].X
    Y_st_true = _normlog_with_obs_sf(st_obs, st_hold, target_sum=target_sum_fold)

    st_shared = _normlog_with_obs_sf(st_obs, ST_mask[:, genes].X,       target_sum=target_sum_fold)
    ps_shared = _normlog_with_obs_sf(ps_obs, pseudo_full_k[:, genes].X, target_sum=target_sum_fold)

    # imputation graph 
    X_pseudo_unshared_torch = torch.as_tensor(Y_ps_unshared, device=device, dtype=storm.H.dtype)
    B_all, gene_graph = cal_B_all_and_gene_graph(
        Q_list=storm.Q_list,
        H=storm.H,
        w_list=storm.w_list,
        B_shared=storm.B,
        X_pseudo_unshared=X_pseudo_unshared_torch,
        pseudo_index=1,
        lam=B_RIDGE_LAM,
        knn_k=int(GENE_GRAPH_k),
        chunk_size=2048,
    )
    storm.B_all      = B_all
    storm.gene_graph = gene_graph

    idx_st_to_pseudo = knn_pseudo_indices_cosine_torch(
        C_st=C_st, C_ps=C_ps, k=K_GNN, device=device, batch=1024,
    )
    pseudo_expr_all = np.concatenate([ps_shared, Y_ps_unshared], axis=1).astype(np.float32)
    storm.X_knn     = build_X_knn_from_pseudo_expr(
        pseudo_expr_all=pseudo_expr_all,
        idx_st_to_pseudo=idx_st_to_pseudo,
    )
    storm._spot_graph = spot_graph

    # STORM GNN + predict
    storm.fit_imputer(
        Y_shared=st_shared,
        epochs=GNN_EPOCHS,
        batch_size=GNN_BATCH,
        lr=GNN_LR,
        weight_decay=GNN_WD,
        gnn_layers=GNN_LAYERS,
        mlp_hidden=GNN_HIDDEN,
        sim_weight=GNN_SIM_WEIGHT,
    )

    Y_hat_raw = storm.predict_imputation(batch_size=128)

    Y_hat_final          = _mix_with_knn_mean(Y_hat_all=Y_hat_raw, X_knn=storm.X_knn, alpha=alpha)
    Y_st_pred_holdout    = Y_hat_final[:, g_s:]

    # Metrics
    m = metrics(Y_st_true, Y_st_pred_holdout)
    print(f"gene-wise cosine: {m['gene_wise_cosine_median']:.4f}")
    print(f"gene-wise mse:    {m['gene_wise_mse_median']:.6f}")
    print(f"cell-wise cosine: {m['cell_wise_cosine_median']:.4f}")
    print(f"cell-wise mse:    {m['cell_wise_mse_median']:.6f}")

    storm.free_gpu()
    del spot_graph
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "fold": int(fold_id),
        "n_st_used":       int(ST_mask.n_obs),
        "n_pseudo_used":   int(pseudo_measured.n_obs),
        "n_holdout":       int(g_u),
        "target_sum_fold": float(target_sum_fold),
        "alpha_knn_mix":   float(alpha),
        **m,
    }

# ----------------------------- RUN ------------------------------------------
rows = []
for f, idxs in enumerate(fold_indices):
    holdout = pd.Index([shared_list[i] for i in idxs])
    rows.append(run_one_fold(f, holdout))

df = pd.DataFrame(rows)

print("\n==================== CV SUMMARY ====================")
print(df)

summary = {
    "fold": -1,
    "n_st_used":        int(df["n_st_used"].median()),
    "n_pseudo_used":    int(df["n_pseudo_used"].median()),
    "n_holdout":        int(df["n_holdout"].sum()),
    "target_sum_fold":  float(df["target_sum_fold"].median()),
    "alpha_knn_mix":    float(df["alpha_knn_mix"].mean()),
    "gene_wise_cosine_median": float(df["gene_wise_cosine_median"].mean()),
    "gene_wise_mse_median":    float(df["gene_wise_mse_median"].mean()),
    "cell_wise_cosine_median": float(df["cell_wise_cosine_median"].mean()),
    "cell_wise_mse_median":    float(df["cell_wise_mse_median"].mean()),
}
df2 = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

mean_including_minus1 = {
    "gene_wise_cosine_median": float(df2["gene_wise_cosine_median"].mean()),
    "gene_wise_mse_median":    float(df2["gene_wise_mse_median"].mean()),
    "cell_wise_cosine_median": float(df2["cell_wise_cosine_median"].mean()),
    "cell_wise_mse_median":    float(df2["cell_wise_mse_median"].mean()),
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
