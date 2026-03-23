#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

import numpy as np
import scanpy as sc
import pandas as pd
import torch

from STORM.STORM import STORM
from STORM.Utils.DeconvolutionUtils import sharp_smooth_all

# ----------------------------- CONFIG ------------------------------------------
n_top_genes             = 7500
gamma1                  = 0.8
core_rank               = 35
ppi_topk                = 650
K_MNN                   = 15
spot_num                = 6000
min_cell_number_in_spot = 2
max_cell_number_in_spot = 15
max_cell_types_in_spot  = 3
RADIUS_INTRA_ST         = 140
metric                  = 'cosine'
SEED                    = 0
P_HI                    = 98
POWER                   = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float64

ppi_csv = "/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/ppi_genes.csv"
ST_PATH = "/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/4.Human_Lymph_Node/ST.h5ad"
SC_PATH = "/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/4.Human_Lymph_Node/scRNA.h5ad"

GC_TARGETS = ["B_GC_DZ", "B_GC_LZ", "B_GC_prePB", "T_CD4+_TfH_GC"]

# ----------------------------- DATA LOADING ------------------------------------
ST_adata = sc.read_h5ad(ST_PATH)
ST_adata.var_names_make_unique()
sc_ref = sc.read_h5ad(SC_PATH)
sc_ref.var_names_make_unique()
sc_ref.obs_names_make_unique()

sc_ref.obs["cell_type"]     = sc_ref.obs["cell_type"].astype("category")
sc_ref.obs["cell_type_idx"] = sc_ref.obs["cell_type"].cat.codes.astype(int)
idx_to_word_celltype        = dict(enumerate(sc_ref.obs["cell_type"].cat.categories))

# ----------------------------- STORM pipeline ----------------------------------
storm = STORM(device=device, dtype=dtype, seed=SEED)
storm.adata_st0            = ST_adata
storm.sc_ref               = sc_ref
storm.idx_to_word_celltype = idx_to_word_celltype

storm.generate_pseudo_spots(
    spot_num=spot_num, min_cells=min_cell_number_in_spot,
    max_cells=max_cell_number_in_spot, max_cell_types=max_cell_types_in_spot,
)
storm.build_joint_embedding(n_top_genes=n_top_genes, flavor="seurat")
storm.build_spatial_graph_singlecell_reference_integration(
    radius_intra_st=RADIUS_INTRA_ST, k_mnn=K_MNN, mnn_metric=metric
)
storm.preprocess_and_align_singlecell_reference_integration(n_top_genes=n_top_genes)
storm.build_slices_singlecell_reference_integration()
storm.build_gene_graph(ppi_csv, score_threshold=ppi_topk)
storm.fit(rank=core_rank, gamma=gamma1, iters=60)
storm.attach_embeddings(key="X_STORM")

# ----------------------------- Deconvolution -----------------------------------
storm.deconvolve(method="gbdt", emb_key="X_STORM_rownorm", out_key="Y_hat")

# ----------------------------- Sharp + smooth ----------------------------------
celltype_cols = storm.celltype_cols
cols = [f"Y_hat_{ct}" for ct in celltype_cols]
sharp_smooth_all(
    storm.adata_st, cols,
    p_lo=0, p_hi=P_HI, mode='winsor', gamma=POWER,
    smooth=True, n_neighs=8,
)

# ----------------------------- Spatial plots -----------------------------------
for ct in GC_TARGETS:
    sc.pl.spatial(
        storm.adata_st,
        color=f"Y_hat_{ct}_sharp_smooth",
        cmap="magma",
        spot_size=150,
        save=f"show{ct}_distribution.png",
    )