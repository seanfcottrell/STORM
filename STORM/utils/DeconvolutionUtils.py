import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

def filter_rare_celltypes_cells(adata_cells, ct_key, min_frac=0.03):
    """Filter cells whose type has global fraction < min_frac."""
    if ct_key not in adata_cells.obs:
        raise KeyError(f"Cell type key '{ct_key}' not found in adata_cells.obs")
    ct = adata_cells.obs[ct_key].astype("category")
    freq = ct.value_counts(normalize=True)
    keep_types = freq[freq >= float(min_frac)].index
    mask = ct.isin(keep_types).to_numpy()
    out = adata_cells[mask].copy()
    out.obs[ct_key] = out.obs[ct_key].astype("category")
    return out, list(keep_types)

##### kNN regression #####
def deconvolve_lle_from_pseudospots(
    adata_st, adata_ps,
    emb_key="X_parafac2_rownorm",
    k=30,
    reg=1e-3,
    celltype_cols=None,
    out_key="Y_hat_lle"
):
    X_st = np.asarray(adata_st.obsm[emb_key], dtype=np.float32)
    X_ps = np.asarray(adata_ps.obsm[emb_key], dtype=np.float32)

    if celltype_cols is None:
        candidate = [c for c in adata_ps.obs.columns if c != "cell_num"]
        celltype_cols = [c for c in candidate if np.issubdtype(adata_ps.obs[c].dtype, np.number)]
    Y_ps = adata_ps.obs[celltype_cols].to_numpy(dtype=np.float32)

    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X_ps)
    dist, idx = nn.kneighbors(X_st, return_distance=True)

    n_st, d = X_st.shape
    Y_hat = np.zeros((n_st, Y_ps.shape[1]), dtype=np.float32)
    recon_err = np.zeros(n_st, dtype=np.float32)

    ones = np.ones((k, 1), dtype=np.float32)

    for i in range(n_st):
        x = X_st[i]                 # (d,)
        Xn = X_ps[idx[i]]           # (k, d)

        # LLE covariance: C = (Xn - x)(Xn - x)^T
        Z = Xn - x[None, :]
        C = Z @ Z.T
        # ridge for stability
        C.flat[::k+1] += reg * np.trace(C) / k + 1e-8

        # solve C w = 1, then normalize to sum 1
        w = np.linalg.solve(C, ones).reshape(-1)
        w = w / (w.sum() + 1e-8)

        # enforce nonnegativity for compositional mixing
        w = np.clip(w, 0.0, None)
        w = w / (w.sum() + 1e-8)

        y = (w[:, None] * Y_ps[idx[i]]).sum(axis=0)
        y = np.clip(y, 0.0, None)
        y = y / (y.sum() + 1e-8)
        Y_hat[i] = y

    adata_st.obsm[out_key] = Y_hat
    adata_st.uns[f"{out_key}_celltypes"] = list(celltype_cols)
    return Y_hat

##### GBDT #####
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (ez.sum(axis=1, keepdims=True) + 1e-8)

def clr_transform(Y, eps=1e-4):
    Y = np.clip(Y, 0.0, None)
    Y = Y / (Y.sum(axis=1, keepdims=True) + 1e-8)
    L = np.log(Y + eps)
    return L - L.mean(axis=1, keepdims=True)

def deconvolve_gbdt_from_pseudospots(
    adata_st, adata_ps,
    emb_key="X_parafac2_rownorm",
    celltype_cols=None,
    out_key="Y_hat_gbdt",
    eps=1e-4,
    **hgb_params
):
    X_st = np.asarray(adata_st.obsm[emb_key])
    X_ps = np.asarray(adata_ps.obsm[emb_key])

    if celltype_cols is None:
        candidate = [c for c in adata_ps.obs.columns if c != "cell_num"]
        celltype_cols = [c for c in candidate if np.issubdtype(adata_ps.obs[c].dtype, np.number)]
    Y_ps = adata_ps.obs[celltype_cols].to_numpy(dtype=np.float32)

    Z_ps = clr_transform(Y_ps, eps=eps)

    base = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        l2_regularization=1e-3,
        **hgb_params
    )
    model = MultiOutputRegressor(base, n_jobs=1)
    model.fit(X_ps, Z_ps)

    Z_hat = model.predict(X_st).astype(np.float32)
    Y_hat = softmax(Z_hat).astype(np.float32)

    adata_st.obsm[out_key] = Y_hat
    adata_st.uns[f"{out_key}_celltypes"] = list(celltype_cols)
    return Y_hat
