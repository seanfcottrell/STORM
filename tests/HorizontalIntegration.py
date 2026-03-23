#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

import torch
import scanpy as sc

from STORM.STORM import STORM

# ----------------------------- CONFIG ------------------------------------------
n_top_genes  = 7500
radius_inter = 150
gamma1       = 0.1
core_rank    = 30
alpha        = 0.1
ppi_topk     = 650

section_ids = ['mouse_anterior', 'mouse_posterior']
ppi_csv     = '/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/ppi_genes.csv'
data_dir    = '/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/10.Mouse_Brain_Merge_Anterior_Posterior_Section_2/'

device = "cpu"
dtype  = torch.float64

# ----------------------------- STORM pipeline ----------------------------------
storm = STORM(device=device, dtype=dtype, seed=0)

storm.load_and_preprocess(
    [f'{data_dir}{sid}.h5ad' for sid in section_ids],
    section_ids=section_ids,
    n_top_genes=n_top_genes,
)
storm.align_genes()
storm.build_slices()
storm.run_paste(alpha=alpha)
storm.build_spatial_graph_multislice_integration(radius_intra=150, radius_inter=radius_inter)
storm.build_gene_graph(ppi_csv, score_threshold=ppi_topk)
storm.fit(rank=core_rank, gamma=gamma1, iters=30)
storm.attach_embeddings(key="X_STORM")

# ----------------------------- CLUSTERING -------------------------------------
storm.cluster_leiden(
    n_neighbors=10,
    resolution=1.0,
    key_added="STORM_leiden",
    emb_key="X_STORM",
    per_slice=False,
)

# ----------------------------- PLOT -------------------------------------------
sc.pl.spatial(
    storm._adata_concat,
    color="STORM_leiden",
    spot_size=100,
    save=f"{section_ids[-1]}_clusters.png",
)
