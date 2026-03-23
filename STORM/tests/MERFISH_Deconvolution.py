#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

import torch
import numpy as np
import scanpy as sc
import squidpy as sq

from STORM.STORM import STORM
from STORM.utils import make_synthetic_spots_from_single_cells
from STORM.Utils.DeconvolutionUtils import filter_rare_celltypes_cells
from STORM.Metrics import evaluate_deconvolution

# ----------------------------- CONFIG ------------------------------------------
n_top_genes             = 7500
gamma1                  = 0.5
core_rank               = 30
ppi_topk                = 650
BIN_SIZE                = 0.025
K_MNN                   = 20
spot_num                = 2000
min_cell_number_in_spot = 2
max_cell_number_in_spot = 15
K_DECONV                = 10
max_cell_types_in_spot  = 3
RADIUS_INTRA_ST         = 0.02
metric                  = 'cosine'
SEED                    = 0
RARE_CT_FRAC            = 0.03
MIN_CELLS_PER_SYN_SPOT  = 2

device = "cpu"
dtype  = torch.float64
ppi_csv = "/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/PARAFAC2/ppi_genes.csv"

CT_MAPPING = {
    "Endothelial 1": "Endothelial", "Endothelial 2": "Endothelial",
    "Endothelial 3": "Endothelial",
    "OD Mature 1":   "OD Mature",   "OD Mature 2":   "OD Mature",
    "OD Mature 3":   "OD Mature",   "OD Mature 4":   "OD Mature",
    "OD Immature 1": "OD Immature", "OD Immature 2": "OD Immature",
    "OD Immature 3": "OD Immature", "OD Immature 4": "OD Immature",
}

# ----------------------------- MAIN --------------------------------------------
ST_adata = sq.datasets.merfish()
ST_adata.var_names_make_unique()
slices = ST_adata.obs["Bregma"].unique()

for sl in slices:
    print("\n" + "=" * 80)
    print(f"Processing slice (Bregma) = {sl}")
    print("=" * 80)

    # Cell-type binning + rare CT filter
    adata_cells = ST_adata[ST_adata.obs.Bregma == sl].copy()
    adata_cells.obs["cell_type_binned"] = (
        adata_cells.obs["Cell_class"].astype(str).replace(CT_MAPPING)
    )
    adata_cells, kept_celltypes = filter_rare_celltypes_cells(
        adata_cells, 'cell_type_binned', min_frac=RARE_CT_FRAC
    )
    print(
        f"[rare-ct] kept {len(kept_celltypes)} cell types "
        f"(>= {RARE_CT_FRAC*100:.1f}%), cells now {adata_cells.n_obs}"
    )

    # Synthetic ST spots
    adata_syn = make_synthetic_spots_from_single_cells(
        adata_cells, bin_size=BIN_SIZE,
        min_cells_per_spot=MIN_CELLS_PER_SYN_SPOT,
        layer=None, cell_type_key='cell_type_binned',
    )
    adata_syn = adata_syn[adata_syn.obs["n_cells"].to_numpy() >= MIN_CELLS_PER_SYN_SPOT].copy()
    print(f"[syn] bin_size={BIN_SIZE}  spots={adata_syn.n_obs}  genes={adata_syn.n_vars}")

    # Build sc reference
    sc_ref = adata_cells.copy()
    sc_ref.obs["cell_type"]     = sc_ref.obs["cell_type_binned"].astype("category")
    sc_ref.obs["cell_type_idx"] = sc_ref.obs["cell_type"].cat.codes.astype(int)
    idx_to_word                 = dict(enumerate(sc_ref.obs["cell_type"].cat.categories))

    # STORM pipeline
    storm = STORM(device=device, dtype=dtype, seed=SEED)
    storm.adata_st0            = adata_syn
    storm.sc_ref               = sc_ref
    storm.idx_to_word_celltype = idx_to_word

    storm.generate_pseudo_spots(
        spot_num=spot_num,
        min_cells=min_cell_number_in_spot,
        max_cells=max_cell_number_in_spot,
        max_cell_types=max_cell_types_in_spot,
    )
    storm.build_joint_embedding(n_top_genes=n_top_genes)
    storm.build_spatial_graph_singlecell_reference_integration(
        radius_intra_st=RADIUS_INTRA_ST, k_mnn=K_MNN, mnn_metric=metric
    )
    storm.preprocess_and_align_singlecell_reference_integration(n_top_genes=n_top_genes)
    storm.build_slices_singlecell_reference_integration()
    storm.build_gene_graph(ppi_csv, score_threshold=ppi_topk)
    storm.fit(rank=core_rank, gamma=gamma1, iters=30)
    storm.attach_embeddings(key="X_STORM")

    if "Y_true" not in adata_syn.obsm:
        raise KeyError("Y_true missing from adata_syn — check cell_type_key in make_synthetic_spots.")
    storm.adata_st.obsm["Y_true"]          = np.asarray(adata_syn.obsm["Y_true"], dtype=np.float32)
    storm.adata_st.uns["Y_true_celltypes"] = list(adata_syn.uns["Y_true_celltypes"])

    # Deconvolution
    storm.deconvolve(method="lle", emb_key="X_STORM_rownorm", out_key="Y_hat", k_lle=K_DECONV)

    # Evaluate
    metrics = evaluate_deconvolution(storm.adata_st, true_key="Y_true", pred_key="Y_hat")
    print(
        f"[slice={sl}] RMSE={metrics['rmse_spot_mean']:.4f}  "
        f"JSD={metrics['jsd_spot_mean']:.4f}"
    )