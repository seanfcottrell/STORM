#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import scanpy as sc
import anndata
import scipy.sparse as sp
from sklearn.decomposition import PCA

# ─── local modules ─────────────────────────────────────────────────────────
from STORM.utils import _preprocess_and_hvg
from STORM.Utils.TensorConstructionUtils import (
    build_irregular_slices,
    intersect_hvgs_and_align,
)
from STORM.Utils.TensorDecompositionUtils import attach_QHD_embeddings
from STORM.GraphConstructions import ppi_graph, build_L_spatial_irregular_radius_cross
from STORM.parafac2 import fit_parafac2_graph
from STORM.Metrics import (run_mclust_ari,
                           f1_lisi
)
import paste as pst
# ────────────────────────────────────────────────────────────────────────────────

n_top_genes = 7500
radius_inter = 120
gamma1 = 0.5
core_rank = 30
alpha = 0.1
ppi_topk = 650

# ----------------------------- CONFIG ------------------------------------------
section_ids = ['151673', '151674', '151675', '151676']
ppi_csv   = '/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/ppi_genes.csv'


# ----------------------------- LOAD + HVG --------------------------------------
adatas = []
for sid in section_ids:
    ad = sc.read_h5ad(f'/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/SpatialTranscriptomics/{sid}.h5ad')
    ad.var_names_make_unique()
    ad = _preprocess_and_hvg(ad, n_top_genes=n_top_genes)
    ad = ad[~ad.obs['layer'].isna(), :].copy()  # keep observed
    adatas.append(ad)

NS = len(adatas)

# Intersect HVGs across sections (and align order)
adatas_aligned, genes = intersect_hvgs_and_align(adatas)
print(f"[info] Universal n_g = {len(genes)}")

# ----------------------------- TORCH SETUP -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64 

# Build irregular slices in *this* order (this is the canonical stacking order)
adatas_in_order = adatas_aligned
X_list, ns_list, row_slices = build_irregular_slices(
    adatas_in_order, device=device, dtype=dtype
)
# ----------------------------- PASTE TRANSPORTS -------------
# Single linear group of slices in order
layer_groups = [adatas_in_order]
# Pairwise transports along the group using PASTE
all_transports = []
for group in layer_groups:
    transports = []
    for i in range(len(group) - 1):
        A, B = group[i], group[i+1]
        G0 = pst.match_spots_using_spatial_heuristic(A.obsm["spatial"], B.obsm["spatial"], use_ot=True)
        pi = pst.pairwise_align(A, B, alpha=float(alpha), G_init=G0, norm=True, verbose=False)
        transports.append(np.asarray(pi, dtype=float))
    all_transports.append(transports)
# Stack into a common frame; write corrected coords into .obsm["paste_xy"]
for group, transports in zip(layer_groups, all_transports):
    aligned_group = pst.stack_slices_pairwise(group, transports)  
    for ad_src, ad_aln in zip(group, aligned_group):
        xy = np.asarray(ad_aln.obsm["spatial"], dtype=float)      # (n_cells, 2)
        if xy.ndim != 2 or xy.shape[1] < 2 or xy.shape[0] != ad_src.n_obs:
            raise ValueError(f"Bad aligned coords shape: got {xy.shape}, expected ({ad_src.n_obs}, 2)")
        ad_src.obsm["paste_xy"] = xy[:, :2]

# ----------------------------- SPATIAL LAPLACIAN (S x S) -----------------------
coords_corr = [ad.obsm["paste_xy"] for ad in adatas_in_order]
L_s, ns_chk, row_slices_chk = build_L_spatial_irregular_radius_cross(
    adatas=adatas_in_order,
    coords_corr_list=coords_corr,
    radius_intra=175,       # within-slice (native coords)
    radius_inter=radius_inter       # cross-slice (PASTE frame coords)
)
# ----------------------------- GENE LAPLACIAN (n_g x n_g) ----------------------
L_g = ppi_graph(genes, ppi_csv, score_threshold=ppi_topk)

# ----------------------------- device
# --- L_s ---
Ls_coo = L_s.tocoo()
idx = torch.from_numpy(
    np.vstack((Ls_coo.row, Ls_coo.col)).astype(np.int64)
).to(device)
val = torch.from_numpy(Ls_coo.data.astype(np.float64)).to(device)
Ls_torch = torch.sparse_coo_tensor(idx, val, Ls_coo.shape).coalesce()
# --- L_g ---
Lg_coo = sp.coo_matrix(L_g)  
idx_g = torch.from_numpy(
    np.vstack((Lg_coo.row, Lg_coo.col)).astype(np.int64)
).to(device)
val_g = torch.from_numpy(Lg_coo.data.astype(np.float64)).to(device)
Lg_torch = torch.sparse_coo_tensor(idx_g, val_g, Lg_coo.shape).coalesce()

# ----------------------------- FIT STORM Model --------------------------
Q_list, H, B, w_list, Y = fit_parafac2_graph(
    X_list, Lg_torch, Ls_torch, rank=core_rank, gamma=gamma1, iters=30, rho=1.0, device=device, dtype=dtype
)

# ----------------------------- ATTACH PER-CELL QHD EMBEDDINGS ----------------------
attach_QHD_embeddings(adatas_in_order, Q_list, H, w_list,
                      key="X_parafac2", also_store_shape=False, store_normed=True)

# ----------------------------- GLOBAL METRICS ----------------------------------
adata_all = anndata.AnnData.concatenate(
    *adatas_in_order, batch_key='batch',
    batch_categories=[f"batch{i+1}" for i in range(NS)]
)

QHD_joint = np.vstack([ad.obsm["X_parafac2"] for ad in adatas_in_order])
adata_all.obsm['QHD'] = QHD_joint

F1LISI = f1_lisi(
    adata=adata_all,
    batch_key='batch',
    label_key='layer',
    use_rep='QHD',
    n_neighbors_graph=15,
    k0=90,
    include_self=False,
    standardize=False,
    summary='median'
)
print(f"Integration F1-LISI = {F1LISI:.4f}")
for sid, ad in zip(section_ids, adatas_in_order):
    ari = run_mclust_ari(ad, key='X_parafac2')
    print(f'Section {sid} ARI = {ari:.4f}')

### Batch Mixing Visualization
# 1) PCA on Q embedding
Q = np.asarray(adata_all.obsm["QHD"], float)
pca = PCA(n_components=min(50, Q.shape[1]), random_state=42)
adata_all.obsm["QHD_pca"] = pca.fit_transform(Q)
# 2) UMAP 
sc.pp.neighbors(adata_all, use_rep="QHD_pca", n_neighbors=15, random_state=42)
sc.tl.umap(adata_all, random_state=42)
sc.pl.umap(
    adata_all,
    color=["batch", "layer"],
    wspace=0.4,
    size = 20
)
