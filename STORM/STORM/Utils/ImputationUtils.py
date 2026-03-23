from __future__ import annotations

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse as sp

# ============================================================
# 1) Extend B factor to unshared genes via ridge regression

def scipy_sparse_to_torch_coo(A: sp.spmatrix, device, dtype=torch.float32) -> torch.Tensor:
    A = A.tocoo()
    idx = np.vstack([A.row, A.col]).astype(np.int64)   # (2, nnz)
    i = torch.from_numpy(idx).to(device=device, dtype=torch.long)
    v = torch.from_numpy(A.data).to(device=device, dtype=dtype)
    return torch.sparse_coo_tensor(i, v, size=A.shape, device=device, dtype=dtype).coalesce()

@torch.no_grad()
def infer_unshared_gene_factors_ridge(
    Q_list: List[torch.Tensor],
    H: torch.Tensor,
    w_list: List[torch.Tensor],
    B_shared: torch.Tensor,
    X_pseudo_unshared: torch.Tensor,
    *,
    pseudo_index: int = 1,
    lam: float = 1e-2,
) -> torch.Tensor:
    """
    Infer gene-factor rows for unshared genes by ridge regression
      on S = QHD_k

    Args
    ----
    Q_list: list of (n_k x R) tensors. Q_list[pseudo_index] corresponds to pseudo-spots niche frame
    H: (R x R) factor coupling matrix.
    w_list: list of (R,) factor scalings for ST and pseudo-spots
    B_shared: (g_s x R) gene factors 
    X_pseudo_unshared: (n_pseudo x g_u) expression of unshared genes across pseudo-spots
    pseudo_index: which entry corresponds to pseudo-spots.
    lam: ridge penalty λ (must be > 0 for Cholesky stability).

    Returns
    -------
    B_unshared: (g_u x R) estimated gene factors for unshared genes in scRNAseq data
    """
    device = H.device
    dtype = H.dtype

    Qp = Q_list[pseudo_index].to(device=device, dtype=dtype)               # (n_pseudo x R)
    wp = w_list[pseudo_index].to(device=device, dtype=dtype).view(1, -1)   # (1 x R)
    X  = X_pseudo_unshared.to(device=device, dtype=dtype)                  # (n_pseudo x g_u)

    n_pseudo, R = Qp.shape

    S = (Qp @ H) * wp  # (n_pseudo x R)
    # Ridge: W = (S^T S + lam I)^(-1) S^T X
    StS = S.T @ S
    A = StS + lam * torch.eye(R, device=device, dtype=dtype)
    RHS = S.T @ X  # (R x g_u)

    L = torch.linalg.cholesky(A)
    W = torch.cholesky_solve(RHS, L)      # (R x g_u)
    B_unshared = W.T.contiguous()         # (g_u x R)
    return B_unshared

@torch.no_grad()
def _coalesce_max_sparse(
    idx: torch.Tensor,  # (2, E) long
    val: torch.Tensor,  # (E,)  float
    G: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Coalesce duplicate COO entries by taking MAX value per (i,j).
    """
    lin = idx[0] * G + idx[1]  # (E,)
    order = torch.argsort(lin)
    lin = lin[order]
    idx = idx[:, order]
    val = val[order]

    # group by lin
    uniq, inv = torch.unique_consecutive(lin, return_inverse=True)

    if hasattr(torch.Tensor, "scatter_reduce_"):
        out = torch.full((uniq.numel(),), -torch.inf, device=val.device, dtype=val.dtype)
        out.scatter_reduce_(0, inv, val, reduce="amax", include_self=True)
        # recover idx for uniq: take first occurrence in each group
        first = torch.zeros_like(uniq, dtype=torch.long)
        # positions where a new group starts
        starts = torch.cat([torch.tensor([0], device=lin.device), (lin[1:] != lin[:-1]).nonzero(as_tuple=True)[0] + 1])
        first[:] = starts
        idx_u = idx[:, first]
        return idx_u, out

    # fallback: will SUM duplicates
    first = torch.zeros_like(uniq, dtype=torch.long)
    starts = torch.cat([torch.tensor([0], device=lin.device), (lin[1:] != lin[:-1]).nonzero(as_tuple=True)[0] + 1])
    first[:] = starts
    idx_u = idx[:, first]
    out = torch.zeros((uniq.numel(),), device=val.device, dtype=val.dtype)
    out.scatter_add_(0, inv, val)
    return idx_u, out

@torch.no_grad()
def build_gene_cosine_graph(
    B_all: torch.Tensor,
    *,
    k: int = 20,
    remove_self_loops: bool = True,
    symmetric: bool = True,
    chunk_size: int = 2048,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Build a sparse COO adjacency by cosine kNN between rows of B_all (shared and unshared genes).

    - Directed kNN by default (each gene -> k nearest genes).
    - If symmetric=True, we add reverse edges and coalesce duplicates by max.

    Returns:
      adj: (G x G) sparse COO with cosine weights.
    """
    device = B_all.device
    dtype = B_all.dtype
    G, _ = B_all.shape

    if G <= 1:
        idx = torch.empty((2, 0), device=device, dtype=torch.long)
        val = torch.empty((0,), device=device, dtype=dtype)
        return torch.sparse_coo_tensor(idx, val, (G, G), device=device, dtype=dtype).coalesce()

    k_eff = int(min(k, G - 1 if remove_self_loops else G))
    if k_eff <= 0:
        idx = torch.empty((2, 0), device=device, dtype=torch.long)
        val = torch.empty((0,), device=device, dtype=dtype)
        return torch.sparse_coo_tensor(idx, val, (G, G), device=device, dtype=dtype).coalesce()

    # Normalize rows for cosine
    Bn = B_all / (B_all.norm(dim=1, keepdim=True) + eps)

    rows, cols, vals = [], [], []

    for i0 in range(0, G, chunk_size):
        i1 = min(i0 + chunk_size, G)

        # cosine sim block: (block x G)
        sim = Bn[i0:i1] @ Bn.T

        if remove_self_loops:
            r = torch.arange(i1 - i0, device=device)
            c = torch.arange(i0, i1, device=device)
            sim[r, c] = -torch.inf

        topv, topj = torch.topk(sim, k=k_eff, dim=1, largest=True, sorted=False)  # (block x k)

        src = torch.arange(i0, i1, device=device).unsqueeze(1).expand(-1, k_eff)  # (block x k)
        rows.append(src.reshape(-1))
        cols.append(topj.reshape(-1))
        vals.append(topv.reshape(-1))

    row = torch.cat(rows).to(torch.long)
    col = torch.cat(cols).to(torch.long)
    val = torch.cat(vals).to(dtype)

    m = torch.isfinite(val)
    row, col, val = row[m], col[m], val[m]

    if symmetric:
        row2 = torch.cat([row, col], dim=0)
        col2 = torch.cat([col, row], dim=0)
        val2 = torch.cat([val, val], dim=0)
        idx = torch.stack([row2, col2], dim=0)
        idx_u, val_u = _coalesce_max_sparse(idx, val2, G)
        return torch.sparse_coo_tensor(idx_u, val_u, (G, G), device=device, dtype=dtype).coalesce()

    idx = torch.stack([row, col], dim=0)
    return torch.sparse_coo_tensor(idx, val, (G, G), device=device, dtype=dtype).coalesce()


@torch.no_grad()
def cal_B_all_and_gene_graph(
    Q_list: List[torch.Tensor],
    H: torch.Tensor,
    w_list: List[torch.Tensor],
    B_shared: torch.Tensor,               # (g_s x R)
    X_pseudo_unshared: torch.Tensor,      # (n_pseudo x g_u)
    *,
    pseudo_index: int = 1,
    lam: float = 1e-2,
    knn_k: int = 50,
    chunk_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    1) Infer B_unshared via ridge regression
    2) Concatenate gene factors B_all = [B_shared; B_unshared]
    3) Build gene-gene graph via cosine kNN over rows of B_all

    Returns:
      B_all: (g_s + g_u, R)
      adj:   (G x G) sparse COO adjacency
    """
    B_unshared = infer_unshared_gene_factors_ridge(
        Q_list=Q_list,
        H=H,
        w_list=w_list,
        B_shared=B_shared,
        X_pseudo_unshared=X_pseudo_unshared,
        pseudo_index=pseudo_index,
        lam=lam,
    )

    B_all = torch.cat(
        [B_shared.to(B_unshared.device, B_unshared.dtype), B_unshared],
        dim=0
    )

    adj = build_gene_cosine_graph(
        B_all,
        k=knn_k,
        remove_self_loops=True,
        symmetric=True,
        chunk_size=chunk_size,
    )

    return B_all, adj
###########################################
#kNN pseudo-spots in shared latent space for ST spots
###########################################

def knn_pseudo_indices_cosine_torch(C_st: np.ndarray, C_ps: np.ndarray, k: int, device: torch.device, batch: int = 1024):
    """
    Return idx: (n_st, k) indices for pseudo rows, using cosine similarity of expression
    """
    Cst = torch.as_tensor(C_st, device=device, dtype=torch.float32)
    Cps = torch.as_tensor(C_ps, device=device, dtype=torch.float32)
    n_st = Cst.shape[0]
    idx_all = []

    for i0 in range(0, n_st, batch):
        i1 = min(i0 + batch, n_st)
        sim = Cst[i0:i1] @ Cps.T  # (b x n_pseudo)
        _, idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=True)
        idx_all.append(idx)

    return torch.cat(idx_all, dim=0)  # (n_st, k)


def build_X_knn_from_pseudo_expr(
    pseudo_expr_all: np.ndarray,         # (n_pseudo, G)
    idx_st_to_pseudo: torch.Tensor,      # (n_st, k), on device
) -> torch.Tensor:
    """
    Returns X_knn tensor: (n_st, k, G) an ST spot's  pseudo-spot neighborhood 
    gene expression from shared latent space neighbors
    """
    device = idx_st_to_pseudo.device
    Xps = torch.as_tensor(pseudo_expr_all, device=device, dtype=torch.float32)  # (n_pseudo, G)
    X_knn = Xps[idx_st_to_pseudo]  # fancy indexing => (n_st, k, G)
    return X_knn
