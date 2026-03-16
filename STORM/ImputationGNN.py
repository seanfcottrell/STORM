import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ============================================================
# Gene GraphSAGE GNN  

def _coo_remove_self_loops(adj: torch.Tensor) -> torch.Tensor:
    """Remove diagonal entries from a COO sparse adjacency."""
    adj = adj.coalesce()
    idx = adj.indices()
    val = adj.values()
    mask = idx[0] != idx[1]
    idx = idx[:, mask]
    val = val[mask]
    return torch.sparse_coo_tensor(idx, val, adj.shape, device=adj.device, dtype=adj.dtype).coalesce()


def row_normalize_adj(
    adj: torch.Tensor,
    remove_self_loops: bool = True,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Row-normalize adjacency: A <- D^{-1} A.
    """
    if adj.is_sparse:
        A = _coo_remove_self_loops(adj) if remove_self_loops else adj.coalesce()
        idx = A.indices()
        val = A.values()
        n = A.size(0)

        row_sum = torch.zeros(n, device=A.device, dtype=val.dtype)
        row_sum.scatter_add_(0, idx[0], val)
        inv = 1.0 / (row_sum + eps)

        val = val * inv[idx[0]]
        return torch.sparse_coo_tensor(idx, val, A.shape, device=A.device, dtype=A.dtype).coalesce()

    A = adj
    if remove_self_loops:
        A = A.clone()
        A.fill_diagonal_(0.0)
    row_sum = A.sum(dim=1, keepdim=True)
    return A / (row_sum + eps)


def adj_mm(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Matrix multiply for dense or sparse adj with dense x."""
    if adj.is_sparse:
        return torch.sparse.mm(adj, x)
    return adj @ x


class GSBlock(nn.Module):
    """
    Separable GraphSAGE-like layer using BOTH:
      - gene graph (GxG) message passing
      - spatial graph (BxB) message passing across spots

    Input x is packed as (G, B*k_in)
      F[b,g,:] are the k_in neighbor-features for gene g at spatial spot b.

    Compute:
      gene-neigh:   apply Pg to x -> (G, B*k_in), then reshape to (G*B, k_in)
      spatial-neigh: apply Ps to F along spot axis -> (B, G, k_in) -> pack back to (G, B*k_in)

    Update (GraphSAGE-style concat + linear + nonlinearity) with residual + LayerNorm.
    """
    def __init__(self, k_in: int, k_out: int, dropout: float = 0.0):
        super().__init__()
        self.k_in = k_in
        self.k_out = k_out
        self.dropout = nn.Dropout(dropout)

        # concat [self || gene_neigh || spatial_neigh]
        self.weight = nn.Parameter(torch.empty(k_in * 3, k_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.res_proj = None
        if k_in != k_out:
            self.res_proj = nn.Linear(k_in, k_out, bias=False)

        self.norm = nn.LayerNorm(k_out)

    def forward(
        self,
        x: torch.Tensor,                 # (G, B*k_in)
        adj_norm: torch.Tensor,          # gene graph (G,G)
        adj_spatial_norm: Optional[torch.Tensor] = None  # spatial graph (B,B) OR None
    ) -> torch.Tensor:
        # ---- gene neighbor aggregate ----
        neigh_g = adj_mm(adj_norm, x)  # (G, B*k_in)

        G, BK = x.shape
        assert BK % self.k_in == 0, "Second dim must be divisible by k_in."
        B = BK // self.k_in

        # ---- spatial neighbor aggregate via mode-1 product on F (B,G,k_in) ----
        if adj_spatial_norm is not None:
            assert adj_spatial_norm.size(0) == B and adj_spatial_norm.size(1) == B, \
                f"Spatial adj must be (B,B) with B={B}."

            # unpack to F: (B, G, k_in)
            F_t = x.reshape(G, B, self.k_in).permute(1, 0, 2).contiguous()  # (B,G,k)

            # Ms = Ps @ F.reshape(B, G*k) -> (B,G,k) then repack to (G,B*k)
            Ms = adj_mm(adj_spatial_norm, F_t.reshape(B, G * self.k_in)).reshape(B, G, self.k_in)
            neigh_s = Ms.permute(1, 0, 2).contiguous().reshape(G, B * self.k_in)  # (G,B*k)
        else:
            neigh_s = torch.zeros_like(x)

        # ---- fuse (per (gene,spot) vector of length k_in) ----
        x_rs   = x.reshape(G * B, self.k_in)
        ng_rs  = neigh_g.reshape(G * B, self.k_in)
        ns_rs  = neigh_s.reshape(G * B, self.k_in)

        combined = torch.cat([x_rs, ng_rs, ns_rs], dim=1)  # (G*B, 3*k_in)

        out = self.dropout(combined) @ self.weight         # (G*B, k_out)
        out = F.relu(out)

        # residual + norm 
        res = x_rs if self.res_proj is None else self.res_proj(x_rs)  # (G*B,k_out)
        out = self.norm(res + out)

        out = out.reshape(G, B * self.k_out)              # (G, B*k_out)
        return out


class GeneGraphImputer(nn.Module):
    """
    shared gene-gene graph GNN applied to each ST spot

    Input:
      X_knn: (B, k, G)  where X_knn[b, m, g] is expr(g) in m-th pseudo-neighbor of spot b
    Output:
      Y_hat: (B, G)
    Loss:
      unchanged: sim_weight*(spot cosine + gene cosine) + MSE on shared genes only
    """
    def __init__(
        self,
        k_neighbors: int,
        gnn_layers: int = 2,
        mlp_hidden: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.k = k_neighbors

        self.gnn = nn.ModuleList([GSBlock(self.k, self.k, dropout=dropout) for _ in range(gnn_layers)])

        self.head = nn.Sequential(
            nn.Linear(self.k, mlp_hidden * 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden * 2, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

        self.mse = nn.MSELoss(reduction="mean")
        self.cos_row = nn.CosineSimilarity(dim=1)  # per-spot cosine over genes
        self.cos_col = nn.CosineSimilarity(dim=0)  # per-gene cosine over spots

        # will be set during fit_gene_imputer if spatial_graph is provided
        self.adj_spatial_norm: Optional[torch.Tensor] = None

    def forward(self, X_knn: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        X_knn: (B, k, G) or (B*k, G)
        adj_norm: gene graph normalized (G,G)

        Returns: (B, G)
        """
        if X_knn.dim() == 3:
            B, k, G = X_knn.shape
            assert k == self.k, f"Expected k={self.k}, got {k}"
            x = X_knn.permute(2, 0, 1).contiguous().reshape(G, B * k)  # (G, B*k)
            B_out, G_out = B, G

            # spatial mixing requires a matching (B,B) adjacency
            adj_sp = self.adj_spatial_norm
            if adj_sp is not None:
                assert adj_sp.size(0) == B and adj_sp.size(1) == B, \
                    "Spatial adjacency size must match B in X_knn."
        elif X_knn.dim() == 2:
            BK, G = X_knn.shape
            assert BK % self.k == 0, f"First dim must be divisible by k={self.k}"
            B_out, G_out = BK // self.k, G
            x = X_knn.t().contiguous()  # (G, B*k)

            if self.adj_spatial_norm is not None:
                raise ValueError(
                    "Provide X_knn as (B,k,G) full-batch to use spatial graph."
                )
        else:
            raise ValueError("X_knn must have shape (B,k,G) or (B*k,G).")

        for layer in self.gnn:
            x = layer(x, adj_norm, self.adj_spatial_norm)

        x2 = x.reshape(-1, self.k)                          # (G*B, k)
        y = self.head(x2).reshape(G_out, B_out).t()         # (B, G)
        return y

    def loss_shared(
        self,
        Y_hat: torch.Tensor,        # (B, G)
        Y_true: torch.Tensor,       # (B, g_s)
        shared_idx: torch.Tensor,   # (g_s,)
        sim_weight: float = 2.0
    ) -> torch.Tensor:
        """
        loss on shared genes:
          loss = sim_weight*(L_spot + L_gene) + MSE
        where:
          L_spot: mean over spots of (1 - cosine(row))
          L_gene: mean over genes of (1 - cosine(col))
        """
        Yh = Y_hat[:, shared_idx]  # (B, g_s)
        Yt = Y_true

        Yh_c = Yh - Yh.mean()
        Yt_c = Yt - Yt.mean()

        L_spot = 1.0 - self.cos_row(Yh_c, Yt_c).mean()
        L_gene = 1.0 - self.cos_col(Yh_c, Yt_c).mean()
        L_mse = self.mse(Yh, Yt)

        return sim_weight * (L_spot + L_gene) + L_mse


class SpotKnnDataset(Dataset):
    def __init__(self, X_knn: torch.Tensor, Y_shared: torch.Tensor):
        """
        X_knn: (n_s, k, G)
        Y_shared: (n_s, g_s)
        """
        assert X_knn.dim() == 3
        assert Y_shared.dim() == 2
        assert X_knn.size(0) == Y_shared.size(0)
        self.X = X_knn
        self.Y = Y_shared

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def fit_gene_imputer(
    X_knn: torch.Tensor,          # (n_s, k, G)
    Y_shared: torch.Tensor,       # (n_s, g_s)
    graph: torch.Tensor,          # (G, G) sparse COO or dense   (gene graph)
    shared_idx: torch.Tensor,     # (g_s,)
    *,
    spatial_graph: Optional[torch.Tensor] = None,  # (n_s, n_s) sparse COO or dense (spot graph)
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    gnn_layers: int = 2,
    mlp_hidden: int = 256,
    sim_weight: float = 2.0,
    device: str | torch.device = "cuda",
    dropout: float = 0.0,
) -> Tuple[GeneGraphImputer, torch.Tensor]:
    """
    Trains the imputer:
      - predicts all genes
      - loss only on shared genes (same as before)

    Returns:
      model, adj_norm  (adj_norm is gene graph normalized, as before)
    """
    device = torch.device(device)

    X_knn = X_knn.to(device)
    Y_shared = Y_shared.to(device)
    shared_idx = shared_idx.to(device)

    graph = graph.to(device)
    adj_norm = row_normalize_adj(graph, remove_self_loops=True)

    model = GeneGraphImputer(
        k_neighbors=X_knn.size(1),
        gnn_layers=gnn_layers,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
    ).to(device)

    if spatial_graph is not None:
        spatial_graph = spatial_graph.to(device)
        model.adj_spatial_norm = row_normalize_adj(spatial_graph, remove_self_loops=True)

        # full-batch training to make Ps consistent with spot indexing
        batch_size = X_knn.size(0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = DataLoader(
        SpotKnnDataset(X_knn, Y_shared),
        batch_size=batch_size,
        shuffle=(spatial_graph is None), 
        drop_last=False,
    )

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        for xb, yb in loader:
            yhat = model(xb, adj_norm)
            loss = model.loss_shared(yhat, yb, shared_idx, sim_weight=sim_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)

        avg = total / X_knn.size(0)
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"[epoch {ep:03d}] loss={avg:.6f}")

    return model, adj_norm


@torch.no_grad()
def predict_gene_imputer(
    model: GeneGraphImputer,
    X_knn: torch.Tensor,          # (n_s, k, G) or (n_s*k, G)
    adj_norm: torch.Tensor,
    *,
    batch_size: int = 128,
) -> torch.Tensor:
    """
    Returns predictions (n_s, G) for all gene
    """
    model.eval()
    device = next(model.parameters()).device

    if model.adj_spatial_norm is not None:
        assert X_knn.dim() == 3, "With spatial mixing enabled, provide X_knn as (n_s,k,G)."
        X_knn = X_knn.to(device)
        yhat = model(X_knn, adj_norm)
        return yhat.detach().cpu()

    # (n_s*k, G) case
    BK, G = X_knn.shape
    k = model.k
    assert BK % k == 0
    n_s = BK // k

    preds = []
    for i in range(0, n_s, batch_size):
        rows = slice(i * k, min((i + batch_size) * k, BK))
        xb = X_knn[rows].to(device)
        preds.append(model(xb, adj_norm).detach().cpu())
    return torch.cat(preds, dim=0)
