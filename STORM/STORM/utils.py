import numpy as np
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
import sklearn.neighbors
import torch
from typing import List, Tuple
import scipy.sparse
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import anndata as ad

def _preprocess_and_hvg(adata, n_top_genes=3000, flavor='seurat_v3'):
    """Select HVGs on raw counts (seurat_v3), then normalize + log1p."""

    # --- basic QC filters (work on raw counts) ---
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_cells=3)

    # --- HVG selection on RAW counts (seurat_v3 requirement) ---
    adata_raw = adata.copy()
    sc.pp.highly_variable_genes(
        adata_raw,
        flavor=flavor,
        n_top_genes=min(n_top_genes, adata_raw.n_vars),
        subset=False,
        inplace=True,
    )
    # Transfer the HVG flag back to the main object
    adata.var["highly_variable"] = adata_raw.var["highly_variable"]
    for col in ["highly_variable_rank", "means", "variances",
                "variances_norm", "highly_variable_nbatches"]:
        if col in adata_raw.var.columns:
            adata.var[col] = adata_raw.var[col]
    del adata_raw

    # --- NOW normalise + log‑transform ---
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)

    return adata

def _gene_intersection(adatas, use_hvg=True, reference=0):
    if use_hvg:
        sets = [set(ad.var_names[ad.var["highly_variable"].values]) for ad in adatas]
    else:
        sets = [set(ad.var_names) for ad in adatas]
    common = set.intersection(*sets)
    if not common:
        raise ValueError("No common genes across samples (after HVG filtering).")
    ref = adatas[reference].var_names
    genes = [g for g in ref if g in common]
    return genes


def make_synthetic_spots_from_single_cells(
    adata_cells: ad.AnnData,
    bin_size: float = 200.0,
    coord_key: str = "spatial",
    min_cells_per_spot: int = 2,
    layer: str | None = None,
    cell_type_key: str | None = "cell_type",
):
    X = adata_cells.layers[layer] if layer is not None else adata_cells.X
    X = X.tocsr() if sp.issparse(X) else np.asarray(X)

    coords = np.asarray(adata_cells.obsm[coord_key])[:, :2].astype(float)
    x, y = coords[:, 0], coords[:, 1]

    gx = np.floor((x - x.min()) / bin_size).astype(np.int64)
    gy = np.floor((y - y.min()) / bin_size).astype(np.int64)
    bin_ids = np.array([f"{i}_{j}" for i, j in zip(gx, gy)], dtype=object)

    uniq_bins, inv = np.unique(bin_ids, return_inverse=True)
    n_cells = adata_cells.n_obs
    n_bins = len(uniq_bins)

    B = sp.csr_matrix(
        (np.ones(n_cells, dtype=np.float32), (inv, np.arange(n_cells))),
        shape=(n_bins, n_cells),
    )

    cell_counts = np.asarray(B.sum(axis=1)).ravel().astype(np.int64)
    keep = cell_counts >= int(min_cells_per_spot)

    B = B[keep]
    kept_bins = uniq_bins[keep]
    cell_counts = cell_counts[keep]

    spot_X = B @ X
    spot_coords = (B @ coords) / cell_counts[:, None]

    adata_spots = ad.AnnData(X=spot_X, var=adata_cells.var.copy())
    adata_spots.obs_names = pd.Index([f"spot_{b}" for b in kept_bins], name="spot")
    adata_spots.obsm["spatial"] = spot_coords
    adata_spots.obs["n_cells"] = cell_counts

    if cell_type_key is not None and cell_type_key in adata_cells.obs:
        ct = pd.Categorical(adata_cells.obs[cell_type_key])
        codes = ct.codes.astype(np.int64)
        n_ct = len(ct.categories)

        C = sp.csr_matrix(
            (np.ones(n_cells, dtype=np.float32), (np.arange(n_cells), codes)),
            shape=(n_cells, n_ct),
        )

        spot_ct = (B @ C).toarray()
        Y_true = spot_ct / cell_counts[:, None]

        adata_spots.obsm["Y_true"] = Y_true
        adata_spots.uns["Y_true_celltypes"] = list(ct.categories)

    return adata_spots