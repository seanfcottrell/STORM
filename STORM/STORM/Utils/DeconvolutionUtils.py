import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import scipy.sparse as sp


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

    ones = np.ones((k, 1), dtype=np.float32)

    for i in range(n_st):
        x = X_st[i]
        Xn = X_ps[idx[i]]

        Z = Xn - x[None, :]
        C = Z @ Z.T
        C.flat[::k+1] += reg * np.trace(C) / k + 1e-8

        w = np.linalg.solve(C, ones).reshape(-1)
        w = w / (w.sum() + 1e-8)

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


# ──────────────────────────────────────────────────────────────────────
# Post‑processing: sharp + smooth 
# ──────────────────────────────────────────────────────────────────────

def sharp_smooth_obs_score(
    adata,
    col: str,
    *,
    p_lo: float = 0.0,
    p_hi: float = 99.0,
    mode: str = "keep",
    gamma: float = 2.0,
    out_sharp: str | None = None,
    out_smooth: str | None = None,
    smooth: bool = True,
    coord_type: str = "grid",
    n_neighs: int = 6,
    include_self: bool = True,
    G_key: str = "spatial_connectivities",
    dtype=np.float32,
):
    if col not in adata.obs.columns:
        raise KeyError(f"{col} not found in adata.obs")

    if mode not in {"keep", "winsor"}:
        raise ValueError('mode must be "keep" or "winsor"')

    x = pd.to_numeric(adata.obs[col], errors="coerce").to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    finite = np.isfinite(x)
    if finite.sum() == 0:
        q_lo, q_hi = 0.0, 0.0
    else:
        q_lo = np.percentile(x[finite], p_lo)
        q_hi = np.percentile(x[finite], p_hi)

    if mode == "keep":
        y = np.where((x >= q_lo) & (x <= q_hi), x, 0.0)
    else:
        y = np.clip(x, q_lo, q_hi)

    y = np.clip(y, 0.0, None)
    if gamma != 1.0:
        y = np.power(y, gamma)

    if out_sharp is None:
        out_sharp = f"{col}_sharp"
    adata.obs[out_sharp] = y.astype(dtype)

    if smooth:
        if G_key not in adata.obsp:
            import squidpy as sq
            sq.gr.spatial_neighbors(adata, coord_type=coord_type, n_neighs=n_neighs)

        G = adata.obsp[G_key].tocsr()
        if include_self:
            G = G + sp.eye(G.shape[0], format="csr")

        den = np.asarray(G.sum(axis=1)).ravel()
        den = np.clip(den, 1e-12, None)

        y_smooth = (G @ y) / den

        if out_smooth is None:
            out_smooth = f"{out_sharp}_smooth"
        adata.obs[out_smooth] = y_smooth.astype(dtype)

    return q_lo, q_hi


def sharp_smooth_all(
    adata,
    yhat_cols,
    *,
    p_lo=0.0,
    p_hi=99.0,
    mode="keep",
    gamma=2.0,
    smooth=True,
    n_neighs=6,
):
    present = [c for c in yhat_cols if c in adata.obs.columns]
    missing = [c for c in yhat_cols if c not in adata.obs.columns]
    if missing:
        print("Missing columns (skipping):", missing)

    cutoffs = {}
    for c in present:
        q_lo, q_hi = sharp_smooth_obs_score(
            adata, c,
            p_lo=p_lo, p_hi=p_hi, mode=mode, gamma=gamma,
            out_sharp=f"{c}_sharp",
            out_smooth=f"{c}_sharp_smooth",
            smooth=smooth, n_neighs=n_neighs,
        )
        cutoffs[c] = (q_lo, q_hi)
    return cutoffs