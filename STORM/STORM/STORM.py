#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import random
import gc
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import scanpy as sc
import anndata
from sklearn.decomposition import PCA, TruncatedSVD

# --- STORM internals ---
from STORM.utils import _preprocess_and_hvg
from STORM.Utils.TensorConstructionUtils import build_irregular_slices, intersect_hvgs_and_align
from STORM.Utils.TensorDecompositionUtils import attach_QHD_embeddings
from STORM.Utils.DeconvolutionUtils import (
    deconvolve_gbdt_from_pseudospots,
    deconvolve_lle_from_pseudospots,
    sharp_smooth_all,
)
from STORM.Utils.PseudoSpotUtils import pseudo_spot_generation
from STORM.Utils.ImputationUtils import (
    knn_pseudo_indices_cosine_torch,
    build_X_knn_from_pseudo_expr,
    cal_B_all_and_gene_graph,
    scipy_sparse_to_torch_coo,
)
from STORM.ImputationGNN import fit_gene_imputer, predict_gene_imputer
from STORM.GraphConstructions import (
    ppi_graph,
    build_L_spatial_irregular_radius_cross,
    build_L_st_with_pseudo_mnn,
    _intra_adj_from_coords,
)
from STORM.fit_STORM import fit_STORM

try:
    import paste as pst
    _PASTE_AVAILABLE = True
except ImportError:
    _PASTE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scipy_to_torch_sparse(L: sp.spmatrix, device: torch.device, dtype=torch.float64) -> torch.Tensor:
    coo = L.tocoo()
    idx = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)).to(device)
    val = torch.from_numpy(coo.data.astype(np.float64)).to(device)
    return torch.sparse_coo_tensor(idx, val, coo.shape, dtype=dtype, device=device).coalesce()


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# STORM class
# ---------------------------------------------------------------------------

class STORM:

    def __init__(
        self,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float64,
        seed: int = 0,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.seed = seed
        _set_seeds(seed)

        self.adatas: List[sc.AnnData] = []
        self.section_ids: List[str] = []
        self.adatas_aligned: List[sc.AnnData] = []
        self.genes: pd.Index = pd.Index([])
        self.X_list: List[torch.Tensor] = []
        self.ns_list: List[int] = []

        self.Ls_torch: Optional[torch.Tensor] = None
        self.Lg_torch: Optional[torch.Tensor] = None

        self.Q_list: Optional[List[torch.Tensor]] = None
        self.H: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self.w_list: Optional[List[torch.Tensor]] = None
        self.Y: Optional[torch.Tensor] = None

        self.adata_st: Optional[sc.AnnData] = None
        self.adata_ps: Optional[sc.AnnData] = None
        self.adata_st0: Optional[sc.AnnData] = None
        self.pseudo_spots: Optional[sc.AnnData] = None
        self.sc_ref: Optional[sc.AnnData] = None
        self.idx_to_word_celltype: Dict[int, str] = {}
        self.celltype_cols: List[str] = []

        self.X_knn: Optional[torch.Tensor] = None
        self.gene_graph: Optional[torch.Tensor] = None
        self.B_all: Optional[torch.Tensor] = None

    # -----------------------------------------------------------------------
    # Data loading & preprocessing
    # -----------------------------------------------------------------------

    def load_and_preprocess(
        self,
        paths: List[str],
        section_ids: Optional[List[str]] = None,
        n_top_genes: int = 7500,
        flavor: str = 'seurat_v3',
        domain_key: Optional[str] = None,
        data_dir: str = "",
    ) -> "STORM":
        self.section_ids = section_ids if section_ids is not None else paths
        self.adatas = []
        for p in paths:
            path = f"{data_dir}{p}" if data_dir else p
            ad = sc.read_h5ad(path)
            ad.var_names_make_unique()
            ad = _preprocess_and_hvg(ad, n_top_genes=n_top_genes, flavor=flavor)
            if domain_key is not None and domain_key in ad.obs.columns:
                ad = ad[~ad.obs[domain_key].isna(), :].copy()
            self.adatas.append(ad)
        print(f"[STORM] loaded {len(self.adatas)} slices")
        return self

    def load_st(self, path: str) -> "STORM":
        self.adata_st0 = sc.read_h5ad(path)
        self.adata_st0.var_names_make_unique()
        print(f"[STORM] ST: {self.adata_st0.n_obs} spots × {self.adata_st0.n_vars} genes")
        return self

    def load_sc_reference(
        self,
        path: str,
        label_key: str = "cell_type"
    ) -> "STORM":
        sc_ref = sc.read_h5ad(path)
        sc_ref.var_names_make_unique()
        sc_ref.obs_names_make_unique()
        sc_ref.obs["cell_types"] = sc_ref.obs[label_key].astype("category")
        sc_ref.obs["cell_type_idx"] = sc_ref.obs["cell_types"].cat.codes.astype(int)
        self.idx_to_word_celltype = dict(enumerate(sc_ref.obs["cell_types"].cat.categories))
        self.sc_ref = sc_ref
        print(
            f"[STORM] scRNA ref: {sc_ref.n_obs} cells, "
            f"{len(self.idx_to_word_celltype)} cell types"
        )
        return self

    # -----------------------------------------------------------------------
    # Gene alignment
    # -----------------------------------------------------------------------

    def align_genes(self) -> "STORM":
        self.adatas_aligned, self.genes = intersect_hvgs_and_align(self.adatas)
        print(f"[STORM] aligned genes: {len(self.genes)}")
        return self

    # -----------------------------------------------------------------------
    # Tensor slices
    # -----------------------------------------------------------------------

    def build_slices(self) -> "STORM":
        self.X_list, self.ns_list, _ = build_irregular_slices(
            self.adatas_aligned, device=self.device, dtype=self.dtype
        )
        print(f"[STORM] slice shapes: {[tuple(x.shape) for x in self.X_list]}")
        return self

    # -----------------------------------------------------------------------
    # PASTE alignment
    # -----------------------------------------------------------------------

    def run_paste(self, alpha: float = 0.1) -> "STORM":
        if not _PASTE_AVAILABLE:
            raise ImportError("paste is not installed. Run: pip install paste-bio")
        adatas = self.adatas_aligned
        transports = []
        for i in range(len(adatas) - 1):
            A, B = adatas[i], adatas[i + 1]
            G0 = pst.match_spots_using_spatial_heuristic(
                A.obsm["spatial"], B.obsm["spatial"], use_ot=True
            )
            pi = pst.pairwise_align(A, B, alpha=float(alpha), G_init=G0, norm=True, verbose=False)
            transports.append(np.asarray(pi, dtype=float))
        aligned = pst.stack_slices_pairwise(adatas, transports)
        for ad_src, ad_aln in zip(adatas, aligned):
            xy = np.asarray(ad_aln.obsm["spatial"], dtype=float)
            if xy.ndim != 2 or xy.shape[1] < 2 or xy.shape[0] != ad_src.n_obs:
                raise ValueError(f"Bad PASTE coords shape: {xy.shape}")
            ad_src.obsm["paste_xy"] = xy[:, :2]
        print(f"[STORM] PASTE alignment done for {len(adatas)} slices")
        return self

    # -----------------------------------------------------------------------
    # Spatial graph (cross-slice PASTE-corrected)
    # -----------------------------------------------------------------------

    def build_spatial_graph_multislice_integration(
        self,
        radius_intra: float,
        radius_inter: float,
    ) -> "STORM":
        coords_corr = [ad.obsm["paste_xy"] for ad in self.adatas_aligned]
        L_s, _, _ = build_L_spatial_irregular_radius_cross(
            adatas=self.adatas_aligned,
            coords_corr_list=coords_corr,
            radius_intra=radius_intra,
            radius_inter=radius_inter,
        )
        self.Ls_torch = _scipy_to_torch_sparse(L_s, self.device, self.dtype)
        print(f"[STORM] spatial graph: {L_s.shape}, nnz={L_s.nnz}")
        return self

    # -----------------------------------------------------------------------
    # Pseudo-spot generation
    # -----------------------------------------------------------------------

    def generate_pseudo_spots(
        self,
        spot_num: int = 5000,
        min_cells: int = 2,
        max_cells: int = 15,
        max_cell_types: int = 3,
        method: str = "celltype",
    ) -> "STORM":
        if self.sc_ref is None:
            raise RuntimeError("Call load_sc_reference() before generate_pseudo_spots().")
        self.pseudo_spots = pseudo_spot_generation(
            sc_exp=self.sc_ref,
            idx_to_word_celltype=self.idx_to_word_celltype,
            spot_num=spot_num,
            min_cell_number_in_spot=min_cells,
            max_cell_number_in_spot=max_cells,
            max_cell_types_in_spot=max_cell_types,
            generation_method=method,
        )
        print(f"[STORM] pseudo-spots: {self.pseudo_spots.n_obs}")
        return self

    # -----------------------------------------------------------------------
    # Joint embedding of ST and Pseudospots
    # -----------------------------------------------------------------------

    def build_joint_embedding(
        self,
        n_top_genes: int = 7500,
        n_pcs: int = 50,
        flavor: str = "seurat_v3",
    ) -> "STORM":
        """
        Align ST and pseudo-spot gene spaces, compute a joint PCA embedding.
        """
        if self.adata_st0 is None or self.pseudo_spots is None:
            raise RuntimeError("Call load_st() and generate_pseudo_spots() first.")

        genes0 = self.adata_st0.var_names.intersection(self.pseudo_spots.var_names)
        self._adata_st_genes0 = self.adata_st0[:, genes0].copy()
        self._pseudo_genes0 = self.pseudo_spots[:, genes0].copy()

        adata_joint = anndata.concat(
            {"st": self._adata_st_genes0, "pseudo": self._pseudo_genes0},
            join="inner",
            label="domain",
            index_unique=None,
        ).copy()

        n_top = min(n_top_genes, adata_joint.n_vars)

        if flavor == "seurat_v3":
            adata_raw = adata_joint.copy()
            sc.pp.highly_variable_genes(
                adata_raw,
                n_top_genes=n_top,
                batch_key="domain",
                flavor="seurat_v3",
                inplace=True,
            )
            hvg_mask = adata_raw.var["highly_variable"].copy()
            del adata_raw

            # normalise + log-transform
            sc.pp.normalize_total(adata_joint, target_sum=1e4)
            sc.pp.log1p(adata_joint)

            # Apply the pre-computed HVG mask
            adata_joint.var["highly_variable"] = hvg_mask
        else:
            # For flavors that expect normalised data (e.g. "seurat")
            sc.pp.normalize_total(adata_joint, target_sum=1e4)
            sc.pp.log1p(adata_joint)
            sc.pp.highly_variable_genes(
                adata_joint,
                n_top_genes=n_top,
                batch_key="domain",
                flavor=flavor,
                inplace=True,
            )

        adata_mnn = adata_joint[:, adata_joint.var["highly_variable"]].copy()

        Xj = adata_mnn.X
        n_pcs_eff = min(n_pcs, adata_mnn.n_vars - 1)
        if sp.issparse(Xj):
            self._emb_joint = TruncatedSVD(
                n_components=n_pcs_eff, random_state=self.seed
            ).fit_transform(Xj)
        else:
            self._emb_joint = PCA(
                n_components=n_pcs_eff, random_state=self.seed
            ).fit_transform(np.asarray(Xj, dtype=np.float32))

        print(f"[STORM] joint embedding: {self._emb_joint.shape}")
        return self

    # -----------------------------------------------------------------------
    # Spatial graph (ST + sc pseudospots via MNN)
    # -----------------------------------------------------------------------

    def build_spatial_graph_singlecell_reference_integration(
        self,
        radius_intra_st: float,
        k_mnn: int = 20,
        mnn_metric: str = "cosine",
    ) -> "STORM":
        if not hasattr(self, "_emb_joint"):
            raise RuntimeError("Call build_joint_embedding() first.")
        L_s, self._ns_st, self._rows_st, self._row_pseudo = build_L_st_with_pseudo_mnn(
            st_adatas=[self._adata_st_genes0],
            pseudo_adata=self._pseudo_genes0,
            emb_joint=self._emb_joint,
            radius_intra_st=radius_intra_st,
            k_mnn=k_mnn,
            mnn_metric=mnn_metric,
        )
        self.Ls_torch = _scipy_to_torch_sparse(L_s, self.device, self.dtype)
        print(f"[STORM] MNN spatial graph: {L_s.shape}, nnz={L_s.nnz}")
        return self

    # -----------------------------------------------------------------------
    # Preprocess + align (ST + sc individually)
    # -----------------------------------------------------------------------

    def preprocess_and_align_singlecell_reference_integration(self, n_top_genes: int = 7500) -> "STORM":
        if not hasattr(self, "_adata_st_genes0"):
            raise RuntimeError("Call build_joint_embedding() first.")
        st_p = _preprocess_and_hvg(self._adata_st_genes0.copy(), n_top_genes=n_top_genes)
        ps_p = _preprocess_and_hvg(self._pseudo_genes0.copy(), n_top_genes=n_top_genes)
        self.adatas_aligned, self.genes = intersect_hvgs_and_align([st_p, ps_p])
        self.adata_st = self.adatas_aligned[0]
        self.adata_ps = self.adatas_aligned[1]
        print(f"[STORM] singlecell_reference_integration aligned genes: {len(self.genes)}")
        return self

    # -----------------------------------------------------------------------
    # Gene Laplacian (PPI)
    # -----------------------------------------------------------------------

    def build_gene_graph(
        self,
        ppi_csv: str,
        score_threshold: int = 650,
        case: str = "upper",
    ) -> "STORM":
        L_g = ppi_graph(self.genes, ppi_csv, score_threshold=score_threshold, case=case)
        self.Lg_torch = _scipy_to_torch_sparse(sp.coo_matrix(L_g), self.device, self.dtype)
        print(f"[STORM] gene graph: {L_g.shape}, nnz={sp.coo_matrix(L_g).nnz}")
        return self

    # -----------------------------------------------------------------------
    # Build X_list
    # -----------------------------------------------------------------------

    def build_slices_singlecell_reference_integration(self) -> "STORM":
        self.X_list, self.ns_list, _ = build_irregular_slices(
            self.adatas_aligned, device=self.device, dtype=self.dtype
        )
        return self

    # -----------------------------------------------------------------------
    # Fit STORM
    # -----------------------------------------------------------------------

    def fit(
        self,
        rank: int = 30,
        gamma: float = 0.5,
        iters: int = 50,
        rho: float = 1.0,
        tol: float = 1e-5,
    ) -> "STORM":
        if self.Ls_torch is None or self.Lg_torch is None:
            raise RuntimeError("Build spatial and gene graphs before calling fit().")
        if not self.X_list:
            raise RuntimeError("Build tensor slices before calling fit().")
        self.Q_list, self.H, self.B, self.w_list, self.Y = fit_STORM(
            self.X_list,
            self.Lg_torch,
            self.Ls_torch,
            rank=rank,
            gamma=gamma,
            iters=iters,
            rho=rho,
            tol=tol,
            device=self.device,
            dtype=self.dtype,
            seed=self.seed,
        )
        return self

    # -----------------------------------------------------------------------
    # Attach QHD embeddings
    # -----------------------------------------------------------------------

    def attach_embeddings(
        self,
        key: str = "X_STORM",
        also_store_shape: bool = False,
        store_normed: bool = True,
    ) -> "STORM":
        if self.Q_list is None:
            raise RuntimeError("Call fit() before attach_embeddings().")
        attach_QHD_embeddings(
            self.adatas_aligned,
            self.Q_list, self.H, self.w_list,
            key=key,
            also_store_shape=also_store_shape,
            store_normed=store_normed,
        )
        if len(self.adatas_aligned) >= 2:
            self.adata_st = self.adatas_aligned[0]
            self.adata_ps = self.adatas_aligned[1]
        if (
            self.adata_st is not None
            and self.adata_st0 is not None
            and "spatial" in self.adata_st0.obsm
            and "spatial" not in self.adata_st.obsm
        ):
            self.adata_st.obsm["spatial"] = self.adata_st0.obsm["spatial"].copy()
        print(f"[STORM] embeddings attached (key='{key}')")
        return self

    # -----------------------------------------------------------------------
    # Clustering
    # -----------------------------------------------------------------------

    def cluster_leiden(
        self,
        n_neighbors: int = 10,
        resolution: float = 1.0,
        key_added: str = "STORM_leiden",
        emb_key: str = "X_STORM",
        per_slice: bool = True,
    ) -> "STORM":
        adatas = self.adatas_aligned
        if per_slice:
            for sid, ad in zip(self.section_ids, adatas):
                sc.pp.neighbors(ad, use_rep=emb_key, n_neighbors=n_neighbors)
                sc.tl.leiden(ad, resolution=resolution, key_added=key_added)
                print(f"[STORM] Leiden done for {sid}")
        else:
            adata_concat = anndata.concat(
                adatas, label="section",
                keys=self.section_ids if self.section_ids else None,
                join="inner", merge="same",
            )
            sc.pp.neighbors(adata_concat, use_rep=emb_key, n_neighbors=n_neighbors)
            sc.tl.leiden(adata_concat, resolution=resolution, key_added=key_added)
            self._adata_concat = adata_concat
            print("[STORM] Leiden done on concatenated adata")
        return self

    def cluster_mclust(
        self,
        n_clusters: int = None,
        emb_key: str = "X_STORM",
        model: str = "EEE",
        key_added: str = "STORM_mclust",
        per_slice: bool = True,
    ) -> "STORM":
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        from rpy2.robjects.conversion import localconverter

        base   = importr("base")
        mclust = importr("mclust")

        def _run_mclust(ad, k):
            X = np.asarray(ad.obsm[emb_key], float)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            if k is None:
                raise ValueError("n_clusters must be provided")
            with localconverter(ro.default_converter + numpy2ri.converter):
                Xr = ro.conversion.py2rpy(X)
            base.set_seed(42)
            fit = mclust.Mclust(Xr, G=k, modelNames=model)
            pred = np.array(fit.rx2("classification"), dtype=int).ravel()
            ad.obs[key_added] = pred
            print(f"[STORM] mclust k={k}, unique labels={np.unique(pred)}")

        if per_slice:
            for sid, ad in zip(self.section_ids, self.adatas_aligned):
                _run_mclust(ad, n_clusters)
                print(f"[STORM] mclust clustering done for {sid}")
        else:
            adata_concat = anndata.concat(
                self.adatas_aligned, label="section",
                keys=self.section_ids if self.section_ids else None,
                join="inner", merge="same",
            )
            adata_concat.obsm[emb_key] = np.vstack(
                [ad.obsm[emb_key] for ad in self.adatas_aligned]
            )
            _run_mclust(adata_concat, n_clusters)
            off = 0
            for ad in self.adatas_aligned:
                n = ad.n_obs
                ad.obs[key_added] = adata_concat.obs[key_added].iloc[off:off + n].values
                off += n
            self._adata_concat = adata_concat
            print("[STORM] mclust clustering done on concatenated adata")
        return self

    # -----------------------------------------------------------------------
    # Deconvolution
    # -----------------------------------------------------------------------

    def deconvolve(
        self,
        method: str = "gbdt",
        emb_key: str = "X_STORM_rownorm",
        out_key: str = "Y_hat",
        k_lle: int = 30,
    ) -> "STORM":
        if self.adata_st is None or self.adata_ps is None:
            raise RuntimeError(
                "Call preprocess_and_align_singlecell_reference_integration() "
                "and attach_embeddings() first."
            )
        self.celltype_cols = [
            self.idx_to_word_celltype[i]
            for i in range(len(self.idx_to_word_celltype))
            if self.idx_to_word_celltype[i] in self.adata_ps.obs.columns
        ]

        if method == "gbdt":
            raw_key = f"{out_key}_gbdt"
            deconvolve_gbdt_from_pseudospots(
                self.adata_st, self.adata_ps,
                emb_key=emb_key,
                out_key=raw_key,
                celltype_cols=self.celltype_cols,
            )
            Y_hat = np.asarray(self.adata_st.obsm[raw_key], dtype=np.float32)
        elif method == "lle":
            raw_key = f"{out_key}_lle"
            deconvolve_lle_from_pseudospots(
                self.adata_st, self.adata_ps,
                emb_key=emb_key,
                k=k_lle,
                out_key=raw_key,
                celltype_cols=self.celltype_cols,
            )
            Y_hat = np.asarray(self.adata_st.obsm[raw_key], dtype=np.float32)
        else:
            raise ValueError(f"method must be 'gbdt' or 'lle', got '{method}'")

        self.adata_st.obsm[out_key] = Y_hat
        self.adata_st.uns[f"{out_key}_celltypes"] = list(self.celltype_cols)

        for j, ct in enumerate(self.celltype_cols):
            self.adata_st.obs[f"{out_key}_{ct}"] = Y_hat[:, j]

        print(f"[STORM] deconvolution done ({method}), {len(self.celltype_cols)} cell types")
        return self

    # -----------------------------------------------------------------------
    # Post-processing: smooth deconvolution
    # -----------------------------------------------------------------------

    def smooth_deconvolution(
        self,
        out_key: str = "Y_hat",
        p_lo: float = 0.0,
        p_hi: float = 99.9,
        mode: str = "keep",
        gamma: float = 2.0,
        smooth: bool = True,
        n_neighs: int = 6,
    ) -> Dict:
        if self.adata_st is None:
            raise RuntimeError("Run deconvolve() first.")
        celltypes = list(self.adata_st.uns.get(f"{out_key}_celltypes", self.celltype_cols))
        yhat_cols = [f"{out_key}_{ct}" for ct in celltypes]
        cutoffs = sharp_smooth_all(
            self.adata_st, yhat_cols,
            p_lo=p_lo, p_hi=p_hi, mode=mode, gamma=gamma,
            smooth=smooth, n_neighs=n_neighs,
        )
        print(f"[STORM] smoothing done for {len(cutoffs)} cell types")
        return cutoffs

    # -----------------------------------------------------------------------
    # Gene imputation
    # -----------------------------------------------------------------------

    def build_imputation_graph(
        self,
        imp_genes: List[str],
        X_pseudo_unshared: np.ndarray,
        lam: float = 1.0,
        knn_k: int = 20,
        k_gnn: int = 50,
        radius_intra_st: Optional[float] = None,
    ) -> "STORM":
        if self.Q_list is None or self.B is None:
            raise RuntimeError("Call fit() first.")
        if self.adata_st is None or self.adata_ps is None:
            raise RuntimeError("Call preprocess_and_align_sc_reference() and attach_embeddings() first.")

        X_ps_unshared_t = torch.as_tensor(
            X_pseudo_unshared, device=self.device, dtype=self.H.dtype
        )
        self.B_all, self.gene_graph = cal_B_all_and_gene_graph(
            Q_list=self.Q_list, H=self.H, w_list=self.w_list,
            B_shared=self.B, X_pseudo_unshared=X_ps_unshared_t,
            pseudo_index=1, lam=lam, knn_k=knn_k,
        )

        C_st = np.asarray(self.adata_st.obsm["X_STORM_rownorm"], dtype=np.float32)
        C_ps = np.asarray(self.adata_ps.obsm["X_STORM_rownorm"], dtype=np.float32)

        idx_st_to_pseudo = knn_pseudo_indices_cosine_torch(
            C_st=C_st, C_ps=C_ps, k=k_gnn, device=self.device
        )
        g_s = len(self.genes)
        ps_shared = np.asarray(
            self.adata_ps.X.toarray() if sp.issparse(self.adata_ps.X) else self.adata_ps.X,
            dtype=np.float32,
        )
        pseudo_expr_all = np.concatenate([ps_shared, X_pseudo_unshared], axis=1).astype(np.float32)
        self.X_knn = build_X_knn_from_pseudo_expr(
            pseudo_expr_all=pseudo_expr_all,
            idx_st_to_pseudo=idx_st_to_pseudo,
        )

        if radius_intra_st is not None and "spatial" in self.adata_st.obsm:
            coords = np.asarray(self.adata_st.obsm["spatial"])[:, :2]
            A_sp = _intra_adj_from_coords(coords, radius=radius_intra_st)
            self._spot_graph = scipy_sparse_to_torch_coo(A_sp, self.device, dtype=torch.float32)
        else:
            self._spot_graph = None

        print(
            f"[STORM] imputation graph built — "
            f"G_all={self.B_all.shape[0]}, X_knn={tuple(self.X_knn.shape)}"
        )
        return self

    def fit_imputer(
        self,
        Y_shared: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        gnn_layers: int = 1,
        mlp_hidden: int = 128,
        sim_weight: float = 0.5,
    ) -> "STORM":
        if self.X_knn is None or self.gene_graph is None:
            raise RuntimeError("Call build_imputation_graph() first.")
        g_s = len(self.genes)
        Y_shared_t = torch.as_tensor(Y_shared, device=self.device, dtype=torch.float32)
        shared_idx = torch.arange(g_s, device=self.device, dtype=torch.long)
        gene_graph = self.gene_graph.to(self.device).float()
        self._imputer_model, self._adj_norm = fit_gene_imputer(
            X_knn=self.X_knn, Y_shared=Y_shared_t, graph=gene_graph,
            shared_idx=shared_idx,
            spatial_graph=getattr(self, "_spot_graph", None),
            epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, gnn_layers=gnn_layers,
            mlp_hidden=mlp_hidden, sim_weight=sim_weight, device=self.device,
        )
        return self

    def predict_imputation(self, batch_size: int = 128) -> np.ndarray:
        if not hasattr(self, "_imputer_model"):
            raise RuntimeError("Call fit_imputer() first.")
        Y_hat = predict_gene_imputer(
            self._imputer_model, self.X_knn, self._adj_norm, batch_size=batch_size
        ).detach().cpu().numpy()
        return Y_hat

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def free_gpu(self) -> "STORM":
        for attr in ("X_list", "Ls_torch", "Lg_torch", "gene_graph", "X_knn",
                     "Q_list", "H", "B", "w_list", "Y", "B_all"):
            setattr(self, attr, None)
        if hasattr(self, "_imputer_model"):
            del self._imputer_model
        if hasattr(self, "_adj_norm"):
            del self._adj_norm
        torch.cuda.empty_cache()
        gc.collect()
        print("[STORM] GPU tensors released")
        return self