import numpy as np
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
import sklearn.neighbors
import scanpy as sc
import torch
from typing import List, Tuple
import scipy.sparse
from scipy.sparse import issparse
import matplotlib.pyplot as plt

def intersect_hvgs_and_align(adatas: List) -> Tuple[List, pd.Index]:
    """
    Intersect HVGs across adatas and align columns to a common order
    (preserve the order from the first AnnData).
    """
    # Ensure HVGs exist
    for i, ad in enumerate(adatas):
        if 'highly_variable' not in ad.var.columns:
            raise ValueError(f"adata[{i}] missing 'highly_variable'; run HVG selection first.")
    # Strict intersection of HVG names
    hvg_sets = [set(ad.var_names[ad.var['highly_variable']]) for ad in adatas]
    hvg_inter = set.intersection(*hvg_sets)
    if not hvg_inter:
        raise ValueError("Empty HVG intersection across datasets.")
    # Keep only genes present in all adatas
    present_in_all = set.intersection(*[set(ad.var_names) for ad in adatas])
    keep = hvg_inter & present_in_all
    if not keep:
        raise ValueError("No common genes remain after presence check.")
    # Preserve order from the first AnnData
    genes = pd.Index([g for g in adatas[0].var_names if g in keep], name="gene")
    # Subset & align gene order
    aligned = [ad[:, genes].copy() for ad in adatas]
    return aligned, genes


def build_irregular_slices(
    adatas_aligned: List[sc.AnnData],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[List[torch.Tensor], List[int], List[slice]]:
    """
    Convert aligned AnnData objects -> list of dense torch tensors [n_s_k, n_g].
    Also returns ns_list and row_slices for global stacking order.
    """
    X_list: List[torch.Tensor] = []
    ns_list: List[int] = []
    row_slices: List[slice] = []

    offset = 0
    for ad in adatas_aligned:
        ns, ng = ad.n_obs, ad.n_vars
        ns_list.append(ns)
        row_slices.append(slice(offset, offset + ns))
        offset += ns

        X_np = ad.X.toarray() if scipy.sparse.issparse(ad.X) else np.asarray(ad.X)
        X_t = torch.as_tensor(X_np, device=device, dtype=dtype).contiguous()
        X_list.append(X_t)

    return X_list, ns_list, row_slices
