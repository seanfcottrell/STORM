from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import os
import sys
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# Clustering ARI and F1-LISI
def clustering_ari(ad, domain_key: str, pred_key: str) -> float:
    obs_df = ad.obs.dropna(subset=[domain_key])
    ari = float(adjusted_rand_score(obs_df[pred_key].loc[obs_df.index],
                                    obs_df[domain_key]))
    print(f"ARI={ari:.4f}")
    return ari
# ----------------------------- iLISI / F1-LISI (graph-based) -------------------
def ilisi_graph(
    adata,
    batch_key: str,
    use_rep: str = "X",
    n_neighbors_graph: int = 15,
    k0: int = 90,
    scale: bool = True,
    summary: str = "median",
    include_self: bool = False,
    standardize: bool = False,
    chunk_size: int = 1024
):
    if batch_key not in adata.obs:
        raise KeyError(f"Missing {batch_key} in adata.obs")
    if use_rep != "X" and use_rep not in adata.obsm:
        raise KeyError(f"Missing embedding {use_rep} in adata.obsm")

    rep_key = use_rep
    tmp_key = None
    if use_rep != "X":
        X = np.asarray(adata.obsm[use_rep], float)
        if standardize:
            X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
        tmp_key = "_tmp_ilisi_rep"
        adata.obsm[tmp_key] = X
        rep_key = tmp_key

    sc.pp.neighbors(adata, n_neighbors=n_neighbors_graph, use_rep=rep_key)
    G = adata.obsp["distances"].tocsr()

    labels = adata.obs[batch_key].astype("category")
    codes  = labels.cat.codes.to_numpy()
    B      = len(labels.cat.categories)
    n      = G.shape[0]

    idx_all = np.arange(n)
    lisi_vals = np.empty(n, dtype=float)

    for start in range(0, n, chunk_size):
        inds = idx_all[start:start+chunk_size]
        D = dijkstra(G, directed=False, indices=inds)
        for r, i in enumerate(inds):
            di = D[r].copy()
            if not include_self:
                di[i] = np.inf
            finite = np.isfinite(di)
            if finite.sum() == 0:
                lisi_vals[i] = 1.0
                continue
            k_eff = min(k0, finite.sum())
            nbr = np.argpartition(di, k_eff-1)[:k_eff]
            cc = np.bincount(codes[nbr], minlength=B)
            p  = cc / cc.sum()
            lisi_vals[i] = 1.0 / max(np.sum(p*p), 1e-12)

    if tmp_key is not None and tmp_key in adata.obsm:
        del adata.obsm[tmp_key]

    if scale:
        lisi_vals = (lisi_vals - 1.0) / (B - 1.0) if B > 1 else np.ones_like(lisi_vals)

    if summary == "median":
        return float(np.median(lisi_vals))
    elif summary == "mean":
        return float(np.mean(lisi_vals))
    elif summary == "none":
        return lisi_vals
    else:
        raise ValueError("summary must be 'median', 'mean', or 'none'")

def f1_lisi(
    adata,
    batch_key: str,
    label_key: str,
    use_rep: str = "X",
    n_neighbors_graph: int = 15,
    k0: int = 90,
    include_self: bool = False,
    standardize: bool = False,
    summary: str = "median",
):
    b_vec = ilisi_graph(
        adata, batch_key=batch_key, use_rep=use_rep,
        n_neighbors_graph=n_neighbors_graph, k0=k0,
        scale=True, summary="none", include_self=include_self, standardize=standardize,
    )
    c_vec = ilisi_graph(
        adata, batch_key=label_key, use_rep=use_rep,
        n_neighbors_graph=n_neighbors_graph, k0=k0,
        scale=True, summary="none", include_self=include_self, standardize=standardize,
    )
    sep_vec = 1.0 - c_vec
    denom   = b_vec + sep_vec
    f1_vec  = np.where(denom > 0, 2.0 * b_vec * sep_vec / denom, 0.0)

    if summary == "median":
        return float(np.median(f1_vec))
    elif summary == "mean":
        return float(np.mean(f1_vec))
    elif summary == "none":
        return f1_vec
    else:
        raise ValueError("summary must be 'median', 'mean', or 'none'")
    
# ---------------------- JSD ---------------------------
def _row_normalize(P, eps=1e-12):
    P = np.asarray(P, float)
    P = np.clip(P, 0.0, None)
    s = P.sum(axis=1, keepdims=True)
    return P / (s + eps)

def jsd_multiclass(P, Q, eps=1e-12):
    """
    Jensen–Shannon divergence per row between P and Q.
    P, Q: (n, K) nonnegative
    """
    P = _row_normalize(P, eps)
    Q = _row_normalize(Q, eps)
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    M = 0.5 * (P + Q)
    kl_PM = np.sum(P * np.log(P / M), axis=1)
    kl_QM = np.sum(Q * np.log(Q / M), axis=1)
    return 0.5 * (kl_PM + kl_QM)  # (n,)

def jsd_binary_per_type(P, Q, eps=1e-12):
    """
    For each cell type k, compute binary JSD between [p,1-p] and [q,1-q],
    averaged over spots
    """
    P = _row_normalize(P, eps)
    Q = _row_normalize(Q, eps)
    n, K = P.shape
    out = np.zeros(K, float)
    for k in range(K):
        p = np.clip(P[:, k], eps, 1.0 - eps)
        q = np.clip(Q[:, k], eps, 1.0 - eps)
        P2 = np.stack([p, 1.0 - p], axis=1)
        Q2 = np.stack([q, 1.0 - q], axis=1)
        out[k] = float(np.mean(jsd_multiclass(P2, Q2, eps=eps)))
    return out

def evaluate_deconvolution(adata_st, true_key="Y_true", pred_key="Y_hat"):
    """
    Returns dict with:
      - per-spot summaries
      - per-type dicts (RMSE, Spearman, binary-JSD)
    """
    if true_key not in adata_st.obsm:
        raise KeyError(f"{true_key} not found in adata_st.obsm")
    if pred_key not in adata_st.obsm:
        raise KeyError(f"{pred_key} not found in adata_st.obsm")

    true_cts = list(adata_st.uns.get(f"{true_key}_celltypes", []))
    pred_cts = list(adata_st.uns.get(f"{pred_key}_celltypes", []))
    if len(true_cts) == 0:
        raise KeyError(f"{true_key}_celltypes not found in adata_st.uns")
    if len(pred_cts) == 0:
        raise KeyError(f"{pred_key}_celltypes not found in adata_st.uns")

    df_true = pd.DataFrame(
        np.asarray(adata_st.obsm[true_key], float),
        index=adata_st.obs_names,
        columns=true_cts,
    )
    df_pred = pd.DataFrame(
        np.asarray(adata_st.obsm[pred_key], float),
        index=adata_st.obs_names,
        columns=pred_cts,
    )

    common = df_true.columns.intersection(df_pred.columns)
    if len(common) == 0:
        raise ValueError("No overlapping cell types between ground truth and prediction.")

    df_true = df_true[common].replace([np.inf, -np.inf], np.nan)
    df_pred = df_pred[common].replace([np.inf, -np.inf], np.nan)

    keep = ~(df_true.isna().any(axis=1) | df_pred.isna().any(axis=1))
    df_true = df_true.loc[keep]
    df_pred = df_pred.loc[keep]

    Y_true = df_true.to_numpy(float)
    Y_pred = df_pred.to_numpy(float)

    # Global RMSE over all entries 
    rmse_global = float(np.sqrt(mean_squared_error(Y_true.ravel(), Y_pred.ravel())))

    # Normalize to compositions
    Q = _row_normalize(Y_true)
    P = _row_normalize(Y_pred)

    n_spots, K = Q.shape
    ct_names = list(df_true.columns)

    # per-spot
    jsd_spot = jsd_multiclass(P, Q)                       # (n_spots,)
    rmse_spot = np.sqrt(np.mean((P - Q) ** 2, axis=1))    # per spot, across types

    rho_spot = []
    for i in range(n_spots):
        r, _ = spearmanr(Q[i, :], P[i, :])
        rho_spot.append(r)
    rho_spot = np.asarray(rho_spot, float)

    # per-type
    rmse_per_type = {}
    rho_per_type = {}
    for k, ct in enumerate(ct_names):
        rmse_per_type[ct] = float(np.sqrt(mean_squared_error(Q[:, k], P[:, k])))
        r, _ = spearmanr(Q[:, k], P[:, k])
        rho_per_type[ct] = float(r)

    jsd_bin_vals = jsd_binary_per_type(P, Q)
    jsd_binary_per_type_dict = {ct: float(jsd_bin_vals[k]) for k, ct in enumerate(ct_names)}

    return {
        "common_celltypes": ct_names,
        "rmse": rmse_global,
        "rmse_spot_mean": float(np.nanmean(rmse_spot)),
        "rmse_spot_median": float(np.nanmedian(rmse_spot)),
        "spearman_spot_mean": float(np.nanmean(rho_spot)),
        "spearman_spot_median": float(np.nanmedian(rho_spot)),
        "jsd_spot_mean": float(np.nanmean(jsd_spot)),
        "jsd_spot_median": float(np.nanmedian(jsd_spot)),
        "rmse_per_type": rmse_per_type,
        "rho_per_type": rho_per_type,
        "jsd_binary_per_type": jsd_binary_per_type_dict,
    }

def per_type_metrics_to_df(metrics: dict) -> pd.DataFrame:
    ct = metrics["common_celltypes"]
    df = pd.DataFrame(index=ct)
    df["rmse"] = pd.Series(metrics["rmse_per_type"])
    df["spearman"] = pd.Series(metrics["rho_per_type"])
    df["jsd_binary"] = pd.Series(metrics["jsd_binary_per_type"])
    return df

def slice_type_means(metrics: dict) -> dict:
    df_ct = per_type_metrics_to_df(metrics)
    return {
        "rmse_type_mean": float(df_ct["rmse"].mean(skipna=True)),
        "spearman_type_mean": float(df_ct["spearman"].mean(skipna=True)),
        "jsd_type_binary_mean": float(df_ct["jsd_binary"].mean(skipna=True)),
        "n_common_celltypes": int(df_ct.shape[0]),
    }
