import torch
from torch import Tensor
from typing import List, Tuple
import numpy as np
import scipy.sparse as sp

# ---------- utilities ----------

def _block_row_slices(ns_list: List[int]) -> List[slice]:
    offs, out = 0, []
    for ns in ns_list:
        out.append(slice(offs, offs + ns))
        offs += ns
    return out

def _procrustes_Q(XW_t: Tensor) -> Tensor:
    U, _, Vh = torch.linalg.svd(XW_t, full_matrices=False)
    return U @ Vh

def _diag_of(FtXB: Tensor) -> Tensor:
    return torch.diag(FtXB)

def _make_sym(A): return 0.5 * (A + A.transpose(-1, -2))

def solve_spd_then_fallback(A, B, jitter0=1e-8, growth=10.0, tries=6):
    A = _make_sym(A)
    n = A.shape[-1]
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    scale = torch.clamp(A.diagonal(dim1=-2, dim2=-1).abs().mean(), min=1.0)
    jitter = float(jitter0) * float(scale)

    for _ in range(tries):
        try:
            L = torch.linalg.cholesky(A + jitter * I)
            return torch.cholesky_solve(B, L)
        except RuntimeError:
            jitter *= growth
    try:
        return torch.linalg.solve(A + jitter * I, B)
    except RuntimeError:
        return torch.linalg.lstsq(A + jitter * I, B).solution

def _extract_sparse_diag_scipy(L: sp.spmatrix) -> np.ndarray:
    """Extract main diagonal from a scipy sparse matrix."""
    return np.asarray(L.diagonal(), dtype=np.float64)


def sylvester_cg_numpy(
    Ls_csr: sp.csr_matrix,       # (S, S) scipy CSR
    Lg_csr: sp.csr_matrix,       # (ng, ng) scipy CSR
    alpha: float,
    gamma: float,
    RHS: np.ndarray,             # (S, ng) numpy array
    Y0: np.ndarray = None,       # (S, ng) warm start
    max_iter: int = 300,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Preconditioned CG for the Sylvester equation, entirely in numpy/scipy.

    Uses scipy CSR @ dense for sparse matmuls (MKL-accelerated on most
    installs), which is 3-10x faster than torch.sparse.mm on CPU.
    """
    S, ng = RHS.shape
    g2 = 2.0 * gamma

    # Diagonal preconditioner: M = alpha + 2*gamma*(diag(Ls)_i + diag(Lg)_j)
    ds = _extract_sparse_diag_scipy(Ls_csr)    # (S,)
    dg = _extract_sparse_diag_scipy(Lg_csr)    # (ng,)
    Mdiag = alpha + g2 * (ds[:, None] + dg[None, :])   # (S, ng)
    Mdiag_inv = 1.0 / (Mdiag + 1e-30)

    # Operator A(Y) = alpha*Y + 2*gamma*(Ls @ Y + Y @ Lg)
    # For Y @ Lg when Lg is sparse: compute as (Lg @ Y.T).T
    # which is scipy CSR @ dense — well optimized.
    def A_op(Y):
        LsY = Ls_csr @ Y              # scipy CSR @ dense → dense
        YLg = (Lg_csr @ Y.T).T        # (Lg @ Y^T)^T = Y @ Lg^T; Lg symmetric so = Y @ Lg
        return alpha * Y + g2 * (LsY + YLg)

    # Initialise
    if Y0 is not None:
        Y = Y0.copy()
    else:
        Y = np.zeros_like(RHS)

    R = RHS - A_op(Y)
    Z = R * Mdiag_inv
    P = Z.copy()
    rz_old = np.sum(R * Z)

    r_norm = np.linalg.norm(R)
    if r_norm <= tol:
        return Y

    for _ in range(max_iter):
        AP = A_op(P)
        pAp = np.sum(P * AP) + 1e-30
        alpha_k = rz_old / pAp

        Y += alpha_k * P
        R -= alpha_k * AP

        r_norm = np.linalg.norm(R)
        if r_norm <= tol:
            break

        Z = R * Mdiag_inv
        rz_new = np.sum(R * Z)
        beta = rz_new / (rz_old + 1e-30)
        P = Z + beta * P
        rz_old = rz_new

    return Y


# ---------- torch sparse CG (kept for GPU path) ----------

def sparse_diag(L: torch.Tensor) -> torch.Tensor:
    LC = L.coalesce()
    r, c = LC.indices()
    v = LC.values()
    d = torch.zeros(L.shape[0], dtype=v.dtype, device=v.device)
    m = (r == c)
    if m.any():
        d.index_add_(0, r[m], v[m])
    return d


def sylvester_cg_torch(Ls: Tensor, Lg: Tensor, alpha: float, gamma: float,
                       RHS: Tensor, Y0: Tensor = None, max_iter: int = 300,
                       tol: float = 1e-6) -> Tensor:
    """torch-based CG (used when device is CUDA)."""
    S, ng = RHS.shape

    ds = sparse_diag(Ls) if Ls.is_sparse else torch.diag(Ls)
    dg = sparse_diag(Lg) if Lg.is_sparse else torch.diag(Lg)

    Mdiag = alpha + 2.0 * gamma * (ds[:, None] + dg[None, :])

    def A(Y):
        t1 = torch.sparse.mm(Ls, Y) if Ls.is_sparse else Ls @ Y
        t2 = (torch.sparse.mm(Lg, Y.T).T) if Lg.is_sparse else (Y @ Lg)
        return alpha * Y + 2.0 * gamma * (t1 + t2)

    def M_inv(R):
        return R / (Mdiag + 1e-30)

    Y = torch.zeros_like(RHS) if Y0 is None else Y0.clone()
    R = RHS - A(Y)
    Z = M_inv(R)
    P = Z.clone()
    rz_old = torch.sum(R * Z)

    if torch.linalg.norm(R) <= tol:
        return Y

    for _ in range(max_iter):
        AP = A(P)
        denom = torch.sum(P * AP) + 1e-30
        alpha_k = rz_old / denom
        Y = Y + alpha_k * P
        R = R - alpha_k * AP
        if torch.linalg.norm(R) <= tol:
            break
        Z = M_inv(R)
        rz_new = torch.sum(R * Z)
        beta = rz_new / (rz_old + 1e-30)
        P = Z + beta * P
        rz_old = rz_new
    return Y


# ──────────────────────────────────────────────────────────────────────
# Helpers: torch sparse <-> scipy CSR conversion
# ──────────────────────────────────────────────────────────────────────

def _torch_sparse_to_scipy_csr(T: Tensor) -> sp.csr_matrix:
    """Convert a coalesced torch sparse COO tensor to scipy CSR."""
    T = T.coalesce().cpu()
    idx = T.indices().numpy()
    val = T.values().numpy()
    return sp.coo_matrix((val, (idx[0], idx[1])), shape=T.shape).tocsr()

@torch.no_grad()
def fit_STORM(
    X_list: List[Tensor],
    Lg: Tensor,
    Ls: Tensor,
    rank: int,
    gamma: float,
    iters: int = 50,
    rho: float = 1.0,
    tol: float = 1e-5,
    device: torch.device = None,
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
) -> Tuple[List[Tensor], Tensor, Tensor, List[Tensor], Tensor]:
    """
    Returns: (Q_list, H, B, w_list, Y)
    """
    torch.manual_seed(seed)

    if device is None:
        device = X_list[0].device

    X_list = [x.to(device=device, dtype=dtype) for x in X_list]
    Lg = Lg.to(device=device, dtype=dtype)
    Ls = Ls.to(device=device, dtype=dtype)

    use_cpu_cg = (device.type == "cpu")

    # Pre-convert Laplacians to scipy CSR for numpy CG path
    if use_cpu_cg:
        if Ls.is_sparse:
            Ls_csr = _torch_sparse_to_scipy_csr(Ls)
        else:
            Ls_csr = sp.csr_matrix(Ls.cpu().numpy())
        if Lg.is_sparse:
            Lg_csr = _torch_sparse_to_scipy_csr(Lg)
        else:
            Lg_csr = sp.csr_matrix(Lg.cpu().numpy())
    else:
        # GPU path: coalesce once
        if Ls.is_sparse:
            Ls = Ls.coalesce()
        if Lg.is_sparse:
            Lg = Lg.coalesce()

    nb = len(X_list)
    ns_list = [x.shape[0] for x in X_list]
    ng = X_list[0].shape[1]
    S = sum(ns_list)
    row_chunks = _block_row_slices(ns_list)

    # --- init factors ---
    Q_list = []
    for ns in ns_list:
        G = torch.randn(ns, rank, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(G, mode='reduced')
        Q_list.append(Q[:, :rank].contiguous())
    H = torch.eye(rank, device=device, dtype=dtype)
    B = torch.randn(ng, rank, device=device, dtype=dtype)
    col_norms = torch.clamp(torch.linalg.norm(B, dim=0), min=1e-8)
    B = B / col_norms
    w_list = [torch.ones(rank, device=device, dtype=dtype) * col_norms for _ in range(nb)]

    # ADMM vars
    Y = torch.zeros(S, ng, device=device, dtype=dtype)
    U = torch.zeros_like(Y)
    eyeR = torch.eye(rank, device=device, dtype=dtype)
    epsR = 1e-8

    # numpy buffer for CG warm-start (CPU path only)
    if use_cpu_cg:
        Y_np = np.zeros((S, ng), dtype=np.float64)

    # Main ADMM loop
    for it in range(1, iters + 1):

        # ----- build current S = stack of C_k @ B.T -----
        Yhat_blocks = []
        for k in range(nb):
            F_k = Q_list[k] @ H
            C_k = F_k * w_list[k]
            Yhat_blocks.append(C_k @ B.T)
        S_stack = torch.vstack(Yhat_blocks)

        # ----- Y-step: solve Sylvester via CG -----
        RHS_t = rho * (S_stack - U)

        if use_cpu_cg:
            # Convert to numpy, run scipy CG, convert back
            RHS_np = RHS_t.cpu().numpy().astype(np.float64)
            Y_np = sylvester_cg_numpy(
                Ls_csr, Lg_csr,
                alpha=rho, gamma=gamma,
                RHS=RHS_np, Y0=Y_np,
                max_iter=300, tol=1e-6,
            )
            Y = torch.from_numpy(Y_np).to(device=device, dtype=dtype)
        else:
            # GPU path: use torch sparse CG
            Y = sylvester_cg_torch(
                Ls, Lg,
                alpha=rho, gamma=gamma,
                RHS=RHS_t, Y0=Y,
                max_iter=300, tol=1e-6,
            )

        # primal residual for stopping
        prim_res = torch.linalg.norm(Y - S_stack) / (torch.linalg.norm(Y) + 1e-12)

        # ----- Q-step (Procrustes) -----
        Ytilde = Y + U
        for k in range(nb):
            rows = row_chunks[k]
            Ytilde_k = Ytilde[rows, :]
            W_k = H * w_list[k].unsqueeze(0) @ B.T
            Mk = X_list[k] @ W_k.T + (rho / 2.0) * (Ytilde_k @ W_k.T)
            Q_list[k] = _procrustes_Q(Mk)

        # ----- H-step -----
        SH = torch.zeros(rank, rank, device=device, dtype=dtype)
        RH = torch.zeros(rank, rank, device=device, dtype=dtype)
        for k in range(nb):
            D_k = w_list[k]
            G_k = (B * D_k).T
            SH += G_k @ G_k.T

            QtX = Q_list[k].T @ X_list[k]
            rows = row_chunks[k]
            QtYt = Q_list[k].T @ Ytilde[rows, :]
            RH += (QtX + (rho / 2.0) * QtYt) @ G_k.T

        A_h = SH + epsR * eyeR
        H = solve_spd_then_fallback(A_h, (RH.T / (1.0 + rho / 2.0))).T

        # ----- D-step -----
        BtB = B.T @ B
        for k in range(nb):
            F_k = Q_list[k] @ H
            FtF = F_k.T @ F_k
            Gram = BtB * FtF
            FtX_B = F_k.T @ X_list[k] @ B
            rows = row_chunks[k]
            FtY_B = F_k.T @ Ytilde[rows, :] @ B
            rhs_vec = _diag_of(FtX_B) + (rho / 2.0) * _diag_of(FtY_B)
            A_D = (1.0 + rho / 2.0) * Gram + epsR * eyeR
            w_list[k] = solve_spd_then_fallback(A_D, rhs_vec.unsqueeze(1)).squeeze(1)

        # ----- B-step -----
        SB = torch.zeros(rank, rank, device=device, dtype=dtype)
        RB = torch.zeros(ng, rank, device=device, dtype=dtype)
        for k in range(nb):
            rows = row_chunks[k]
            C_k = (Q_list[k] @ H) * w_list[k]
            SB += C_k.T @ C_k
            RB += X_list[k].T @ C_k
            RB += (rho / 2.0) * (Ytilde[rows, :].T @ C_k)

        A_b = SB + epsR * eyeR
        B = solve_spd_then_fallback(A_b, (RB.T / (1.0 + rho / 2.0))).T

        # column-rescale B → absorb into w_k
        norms = torch.clamp(torch.linalg.norm(B, dim=0), min=1e-8)
        B /= norms
        for k in range(nb):
            w_list[k] = w_list[k] * norms

        # ----- Dual update -----
        Yhat_blocks = []
        for k in range(nb):
            F_k = Q_list[k] @ H
            C_k = F_k * w_list[k]
            Yhat_blocks.append(C_k @ B.T)
        S_stack = torch.vstack(Yhat_blocks)
        U = U + (Y - S_stack)

        if prim_res.item() < tol:
            break

    return Q_list, H, B, w_list, Y
