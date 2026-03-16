import scanpy as sc
import anndata
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
import random
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from scipy.sparse import block_diag as sp_block_diag

############################ Vertical + Horizontal Integration Graphs ###############################

def ppi_graph(genes, ppi_path, score_threshold=700,
                         case="upper"):
    """
    Build a gene Laplacian aligned to the ordered `genes` list.

    Parameters
    ----------
    genes : list-like of str
        The gene names used to build the tensor 
    ppi_path : str
        CSV for PPI interactions with columns: gene1, gene2, combined_score.
    score_threshold : int
        Keep edges with combined_score >= threshold.
    case : {"upper","lower","keep"}
        Case-normalization to match the PPI names

    Returns
    -------
    L : (NG, NG) np.ndarray 
        Unnormalized Laplacian L = D - A for gene relations
    """
    df = pd.read_csv(ppi_path)

    # Pick a consistent case for matching
    if case == "upper":
        g_match = pd.Index([str(g).upper() for g in genes])
        df_g1 = df["gene1"].str.upper()
        df_g2 = df["gene2"].str.upper()
    elif case == "lower":
        g_match = pd.Index([str(g).lower() for g in genes])
        df_g1 = df["gene1"].str.lower()
        df_g2 = df["gene2"].str.lower()
    else:
        g_match = pd.Index([str(g) for g in genes])
        df_g1 = df["gene1"].astype(str)
        df_g2 = df["gene2"].astype(str)

    # Filter by genes and score
    keep = (df["combined_score"] >= score_threshold) & \
           (df_g1.isin(g_match)) & (df_g2.isin(g_match))
    df_sub = df.loc[keep, ["combined_score"]].copy()
    df_sub["g1"] = df_g1[keep].to_numpy()
    df_sub["g2"] = df_g2[keep].to_numpy()

    NG = len(g_match)
    idx = {g: i for i, g in enumerate(g_match)}
    A = np.zeros((NG, NG))

    # Build symmetric adjacency in the *given* gene order
    for (score, g1, g2) in df_sub.itertuples(index=False, name=None):
        i = idx[g1]; j = idx[g2]
        if i == j:
            continue
        val = 1.0 
        if val > A[i, j]:
            A[i, j] = val
            A[j, i] = val

    np.fill_diagonal(A, 0.0)
    D = np.diag(A.sum(axis=1))
    L = (D - A)
    return L

def _intra_adj_from_coords(coords, radius):
    """radius graph within a slice (coords: [n,2])."""
    n = coords.shape[0]
    if n <= 1:
        return sp.csr_matrix((n, n))
    kdt = cKDTree(coords)
    # neighbors within radius on same slice
    D = kdt.sparse_distance_matrix(kdt, max_distance=radius, output_type='coo_matrix')
    # drop self loops 
    keep = D.row != D.col
    rows, cols, d = D.row[keep], D.col[keep], D.data[keep]
    w = np.ones_like(d)
    A = sp.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    A = 0.5 * (A + A.T)
    A.setdiag(0.0); A.eliminate_zeros()
    return A

def _inter_adj_from_radius_cross(coords_corr_list, ns_list, radius):
    """
    Cross-slice radius graph in PASTE-corrected coords 
    coords_corr_list: list of (n_s_k, 2) arrays, all in the same rotated frame
    ns_list: [n_s_1, ..., n_s_B]
    """
    B = len(ns_list)
    offs = np.cumsum([0] + ns_list[:-1])
    S = sum(ns_list)

    rows_all, cols_all, vals_all = [], [], []
    for s in range(B - 1):
        kdt_s = cKDTree(coords_corr_list[s])
        for t in range(s + 1, B):
            kdt_t = cKDTree(coords_corr_list[t])
            # distances for all pairs within radius (s vs t)
            Dst = kdt_s.sparse_distance_matrix(kdt_t, max_distance=radius, output_type='coo_matrix')
            if Dst.nnz == 0:
                continue
            w = np.ones_like(Dst.data, dtype=float)
            
            r_g = offs[s] + Dst.row
            c_g = offs[t] + Dst.col

            # add both directions to keep the graph undirected
            rows_all.extend([*r_g, *c_g])
            cols_all.extend([*c_g, *r_g])
            vals_all.extend([*w,   *w])

    A = sp.coo_matrix((vals_all, (rows_all, cols_all)), shape=(S, S)).tocsr()

    # Symmetrize 
    A = 0.5 * (A + A.T)
    A.setdiag(0.0); A.eliminate_zeros()
    return A

# --- block spatial laplacian ---

def build_L_spatial_irregular_radius_cross(
    adatas,                   # list of AnnData
    coords_corr_list,         # list of (n_s_k, 2) PASTE-corrected coords (same frame)
    radius_intra,             # radius within a slice (native coords)
    radius_inter,             # radius across slices (corrected coords)
):
    """
    Returns:
      L_s: CSR Laplacian on stacked observed nodes (S x S)
      ns:  list of per-slice sizes
      rows: list of slices giving global row ranges
    """
    # sizes + global row slices
    ns = [ad.n_obs for ad in adatas]
    rows = []
    off = 0
    for n in ns:
        rows.append(slice(off, off + n)); off += n
    S = sum(ns)

    # 1) intra-slice adjacency (block diagonal), from each slice's *native* coords
    intra_blocks = []
    for ad in adatas:
        if 'spatial' in ad.obsm:
            coords = np.asarray(ad.obsm['spatial'])[:, :2]
        else:
            # fallback: try obs columns named x/y
            if {'x','y'}.issubset(ad.obs.columns):
                coords = np.c_[ad.obs['x'].to_numpy(), ad.obs['y'].to_numpy()]
            else:
                raise ValueError("Need per-slice spatial coords (ad.obsm['spatial'] or obs['x','y']).")
        A_intra = _intra_adj_from_coords(coords, radius=radius_intra)
        intra_blocks.append(A_intra)
    A_intra_blk = sp.block_diag(intra_blocks, format="csr")

    # 2) inter-slice adjacency from PASTE-corrected coords 
    A_inter = _inter_adj_from_radius_cross(coords_corr_list, ns, radius=radius_inter)

    # 3) combine + Laplacian
    A = A_intra_blk + A_inter
    A.setdiag(0.0); A.eliminate_zeros()
    d = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(d) - A
    return L.tocsr(), ns, rows


############################ Deconvolution + Imputation Graphs ############################ 
def _symmetric_normalize_laplacian(L):
    d = np.asarray(L.diagonal()).astype(float)
    d_inv_sqrt = np.zeros_like(d)
    nz = d > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
    Dinv = sp.diags(d_inv_sqrt)
    return (Dinv @ L @ Dinv).tocsr()

def _mnn_adj_from_embedding(
    emb,
    k=20,
    metric="cosine",   # "cosine" or "euclidean"
):
    """
    Build mutual-NN adjacency in embedding space.

    emb: (N, d) array
    k: number of nearest neighbors (excluding self)
    metric: distance metric for kNN ("cosine", "euclidean", ...)
    """
    N = emb.shape[0]
    if N <= 1:
        return sp.csr_matrix((N, N))

    # kNN in the chosen metric
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric)
    nn.fit(emb)
    dist, idx = nn.kneighbors(emb, return_distance=True)

    # drop self neighbor at position 0
    idx_k = idx[:, 1:k+1]
    dist_k = dist[:, 1:k+1]

    rows = np.repeat(np.arange(N), k)
    cols = idx_k.reshape(-1)
    dvals = dist_k.reshape(-1)

    vals = np.ones_like(dvals, dtype=float)
    
    A_dir = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

    # mutual mask using binary directed graph
    A_bin = A_dir.copy()
    A_bin.data[:] = 1.0
    A_mut_mask = A_bin.minimum(A_bin.T)  # 1 only if both directions exist

    # keep weights only on mutual edges
    A_w = 0.5 * (A_dir + A_dir.T)
    A_mut = A_w.multiply(A_mut_mask)

    A_mut = 0.5 * (A_mut + A_mut.T)

    A_mut.setdiag(0.0)
    A_mut.eliminate_zeros()
    return A_mut.tocsr()

def build_L_st_with_pseudo_mnn(
    st_adatas,
    pseudo_adata,
    emb_joint,
    radius_intra_st,
    k_mnn=20,
    mnn_metric="cosine",
):
    ns_st = [ad.n_obs for ad in st_adatas]
    rows_st = []
    off = 0
    for n in ns_st:
        rows_st.append(slice(off, off + n))
        off += n
    S = sum(ns_st)
    P = pseudo_adata.n_obs
    N = S + P

    if emb_joint.shape[0] != N:
        raise ValueError(f"emb_joint has {emb_joint.shape[0]} rows but expected {N}")

    row_pseudo = slice(S, S + P)

    # --- 1) ST–ST physical adjacency (block diagonal over ST samples) ---
    intra_blocks = []
    for ad in st_adatas:
        if "spatial" in ad.obsm:
            coords = np.asarray(ad.obsm["spatial"])[:, :2]
        elif {"x", "y"}.issubset(ad.obs.columns):
            coords = np.c_[ad.obs["x"].to_numpy(), ad.obs["y"].to_numpy()]
        else:
            raise ValueError("Need ST coords in obsm['spatial'] or obs['x','y'].")

        A_intra = _intra_adj_from_coords(
            coords,
            radius=radius_intra_st
        )
        intra_blocks.append(A_intra)

    A_st_st = sp.block_diag(intra_blocks, format="csr")  # (S,S)

    A_phys_full = sp.block_diag(
        [A_st_st, sp.csr_matrix((P, P), dtype=A_st_st.dtype)],
        format="csr"
    )  # (N,N)

    # --- 2) MNN adjacency on joint embedding (ST and pseudo–pseudo) ---
    k_eff = min(int(k_mnn), max(N - 1, 1))  # safety
    A_mnn_full = _mnn_adj_from_embedding(
        emb_joint,
        k=k_eff,
        metric=mnn_metric,
    )

    # remove ST–ST edges from MNN so ST–ST is purely physical / spatial
    is_st = np.zeros(N, dtype=bool)
    is_st[:S] = True

    A_mnn_coo = A_mnn_full.tocoo()
    keep = ~(is_st[A_mnn_coo.row] & is_st[A_mnn_coo.col])
    A_mnn_masked = sp.coo_matrix(
        (A_mnn_coo.data[keep], (A_mnn_coo.row[keep], A_mnn_coo.col[keep])),
        shape=(N, N)
    ).tocsr()
    A_mnn_masked.setdiag(0.0)
    A_mnn_masked.eliminate_zeros()

    # --- 3) combine adjacency + Laplacian ---
    A = A_phys_full + A_mnn_masked
    A.setdiag(0.0)
    A.eliminate_zeros()

    d = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(d) - A

    return L.tocsr(), ns_st, rows_st, row_pseudo
