#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import scanpy as sc
import anndata
import scipy.sparse as sp
from sklearn.decomposition import PCA
import pandas as pd

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))
from STORM.STORM import STORM
from STORM.Metrics import (clustering_ari, f1_lisi)

# ----------------------------- CONFIG ------------------------------------------
section_ids = ['151673', '151674', '151675', '151676']
ppi_csv   = '/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/ppi_genes.csv'
paths = [f'/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/SpatialTranscriptomics/{sid}.h5ad'
         for sid in section_ids]

# ----------------------------- STORM pipeline ----------------------------------
device = "cpu"
dtype  = torch.float64
storm = STORM(device=device, dtype=dtype, seed=0)

storm.load_and_preprocess(paths, n_top_genes=7500, domain_key="layer")
storm.align_genes()
storm.build_slices()
storm.run_paste(alpha=0.5)
storm.build_spatial_graph_multislice_integration(radius_intra=175, radius_inter=100)
storm.build_gene_graph(ppi_csv, score_threshold=600)
storm.fit(rank=30, gamma=0.5, iters=30)
storm.attach_embeddings()
storm.cluster_mclust(per_slice=True, n_clusters=7)

# ----------------------------- GLOBAL METRICS ----------------------------------
adatas_in_order = storm.adatas_aligned
NS = len(adatas_in_order)

adata_all = anndata.AnnData.concatenate(
    *adatas_in_order, batch_key='batch',
    batch_categories=[f"batch{i+1}" for i in range(NS)]
)
QHD_joint = np.vstack([ad.obsm["X_STORM"] for ad in adatas_in_order])
adata_all.obsm['QHD'] = QHD_joint

F1LISI = f1_lisi(
    adata=adata_all, batch_key='batch', label_key='layer',
    use_rep='QHD', n_neighbors_graph=15, k0=90,
    include_self=False, standardize=False, summary='median'
)
print(f"Integration F1-LISI = {F1LISI:.4f}")

for sid, ad in zip(section_ids, adatas_in_order):
    ari = clustering_ari(ad, 'layer', 'STORM_mclust')
    print(f'Section {sid} ARI = {ari:.4f}')

# ----------------------------- UMAP -------------------------------------------
Q = np.asarray(adata_all.obsm["QHD"], float)
pca = PCA(n_components=min(50, Q.shape[1]), random_state=42)
adata_all.obsm["QHD_pca"] = pca.fit_transform(Q)
sc.pp.neighbors(adata_all, use_rep="QHD_pca", n_neighbors=15, random_state=42)
sc.tl.umap(adata_all, random_state=42)
sc.pl.umap(
    adata_all,
    color=["batch", "layer"],
    wspace=0.4,
    size=20,
    save='DLPFC_UMAP.png'
)
