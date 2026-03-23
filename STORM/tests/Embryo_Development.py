#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

import torch
import scanpy as sc

from STORM.STORM import STORM

# ----------------------------- CONFIG ------------------------------------------
radius_inter = 0.9
radius_intra = 0.9
gamma1       = 0.5
core_rank    = 30
alpha        = 0.1
ppi_topk     = 650

section_ids = ['E10_5_E1S1_MOSTA', 'E11_5_E1S1_MOSTA', 'E12_5_E1S1_MOSTA']
ppi_csv     = '/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/ppi_genes.csv'
scratch_dir = '/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/adatas_with_domains/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float64

# ----------------------------- STORM pipeline ----------------------------------
storm = STORM(device=device, dtype=dtype, seed=0)

storm.load_and_preprocess(
    [f'{scratch_dir}{sid}.h5ad' for sid in section_ids],
    section_ids=section_ids,
    n_top_genes=7000,
)
storm.align_genes()
storm.build_slices()
storm.run_paste(alpha=alpha)
storm.build_spatial_graph_multislice_integration(radius_intra=radius_intra, radius_inter=radius_inter)
storm.build_gene_graph(ppi_csv, score_threshold=ppi_topk)
storm.fit(rank=core_rank, gamma=gamma1, iters=35)
storm.attach_embeddings(key="X_STORM")

# ----------------------------- CLUSTERING + PLOT -------------------------------
storm.cluster_leiden(
    n_neighbors=10,
    resolution=1.2,
    key_added="leiden",
    emb_key="X_STORM",
    per_slice=True,
)

for sid, ad in zip(section_ids, storm.adatas_aligned):
    sc.pl.spatial(
        ad,
        color="leiden",
        img_key=None,
        spot_size=3,
        title=f"{sid} domains",
        save=f"sample_{sid}_domains.png",
    )