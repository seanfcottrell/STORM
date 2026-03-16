import torch
from torch import Tensor
from typing import List, Tuple

# ---------- utilities ----------

def _block_row_slices(ns_list: List[int]) -> List[slice]:
    offs, out = 0, []
    for ns in ns_list:
        out.append(slice(offs, offs + ns))
        offs += ns
    return out

def _procrustes_Q(XW_t: Tensor) -> Tensor:
    # XW_t: [ns_k, R]; return Q_k with orthonormal columns (ns_k x R)
    U, _, Vh = torch.linalg.svd(XW_t, full_matrices=False)
    return U @ Vh

def _diag_of(FtXB: Tensor) -> Tensor:
    return torch.diag(FtXB)

def _make_sym(A): return 0.5 * (A + A.transpose(-1, -2))

def solve_spd_then_fallback(A, B, jitter0=1e-8, growth=10.0, tries=6):
    # solve via cholesky decomposition unless poor conditioning then revert to LU solve or least squares
    A = _make_sym(A)
    n = A.shape[-1]
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    scale = torch.clamp(A.diagonal(dim1=-2, dim2=-1).abs().mean(), min=1.0)
    jitter = float(jitter0) * float(scale)

    for _ in range(tries):
        try:
            L = torch.linalg.cholesky(A + jitter * I)
            return torch.cholesky_solve(B, L)             # cholesky solve for scalability
        except RuntimeError:
            jitter *= growth

    # symmetric indefinite / near-singular
    try:
        return torch.linalg.solve(A + jitter * I, B)       # LU; exact solve if invertible
    except RuntimeError:
        return torch.linalg.lstsq(A + jitter * I, B).solution  # robust least-squares

# ---------- CG for Sylvester equation: (α I + 2γ Ls)Y + 2γ Y Lg = RHS ----------
def sparse_diag(L: torch.Tensor) -> torch.Tensor:
    LC = L.coalesce()
    r, c = LC.indices()
    v = LC.values()
    d = torch.zeros(L.shape[0], dtype=v.dtype, device=v.device)
    m = (r == c)
    if m.any():
        d.index_add_(0, r[m], v[m])
    return d

def sylvester_cg(Ls: Tensor, Lg: Tensor, alpha: float, gamma: float,
                 RHS: Tensor, Y0: Tensor = None, max_iter: int = 300, tol: float = 1e-6) -> Tensor:
    """
    Solve (alpha*I + 2*gamma*Ls) Y + 2*gamma*Y*Lg = RHS.
    """
    S, ng = RHS.shape

    if Ls.is_sparse:
        ds = sparse_diag(Ls)                 
    else:
        ds = torch.diag(Ls)

    if Lg.is_sparse:
        dg = sparse_diag(Lg)
    else:
        dg = torch.diag(Lg)

    Mdiag = alpha + 2.0 * gamma * (ds[:, None] + dg[None, :])  # [S, ng]

    def A(Y: Tensor) -> Tensor:
        t1 = torch.sparse.mm(Ls, Y) if Ls.is_sparse else Ls @ Y
        t2 = (torch.sparse.mm(Lg, Y.T).T) if Lg.is_sparse else (Y @ Lg)
        return alpha * Y + 2.0 * gamma * (t1 + t2)

    def M_inv(R: Tensor) -> Tensor:
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

# ---------- STORM Fit and Transform Data ----------

@torch.no_grad()
def fit_parafac2_graph(
    X_list: List[Tensor],          # list of [n_s_k, n_g] irregular slices
    Lg: Tensor,                    # [n_g, n_g] sparse or dense gene Laplacian (symmetric)
    Ls: Tensor,                    # [S, S]     sparse spatial Laplacian over cells with cross-batch edges
    rank: int,                     # core rank R
    gamma: float,                  # weight on Laplacian regularization
    iters: int = 50,               # AO-ADMM iterations
    rho: float = 1.0,              # ADMM penalty
    tol: float = 1e-5,             # stopping (relative) on Y/S residual
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> Tuple[List[Tensor], Tensor, Tensor, List[Tensor], Tensor]:
    """
    Returns: (Q_list, H, B, w_list, Y)
      - Q_list: list of [n_s_k, R] (orthonormal frames)
      - H: [R, R]
      - B: [n_g, R]
      - w_list: list of [R] (diagonals of D_k)
      - Y: [S, n_g] global reconstruction stack
    """
    torch.manual_seed(seed)
    # Move inputs to device/dtype
    if device is None:
        device = X_list[0].device
    X_list = [x.to(device=device, dtype=dtype) for x in X_list]
    Lg = Lg.to(device=device, dtype=dtype)
    Ls = Ls.to(device=device, dtype=dtype)

    nb = len(X_list)
    ns_list = [x.shape[0] for x in X_list]
    ng = X_list[0].shape[1]
    S = sum(ns_list)
    row_chunks = _block_row_slices(ns_list)

    # --- init factors ---
    # Q_k: random orthonormal (ns_k x R), H: I, B: random then col-norm, w_k: ones
    Q_list = []
    for ns in ns_list:
        G = torch.randn(ns, rank, device=device, dtype=dtype)
        # Orthonormalize via QR
        Q, _ = torch.linalg.qr(G, mode='reduced')  # [ns, R]
        Q_list.append(Q[:, :rank].contiguous())
    H = torch.eye(rank, device=device, dtype=dtype)
    B = torch.randn(ng, rank, device=device, dtype=dtype)
    # column normalize B and absorb into w
    col_norms = torch.clamp(torch.linalg.norm(B, dim=0), min=1e-8)
    B = B / col_norms
    w_list = [torch.ones(rank, device=device, dtype=dtype) * col_norms for _ in range(nb)]

    # ADMM vars
    Y = torch.zeros(S, ng, device=device, dtype=dtype)
    U = torch.zeros_like(Y)
    # Pre-jitter for tiny R systems
    eyeR = torch.eye(rank, device=device, dtype=dtype)
    epsR = 1e-8

    # Main ADMM loop
    if Ls.is_sparse: Ls = Ls.coalesce()
    if Lg.is_sparse: Lg = Lg.coalesce()
    for it in range(1, iters + 1):
        # ----- build current C_k and S -----
        C_list = []
        Yhat_blocks = []
        for k in range(nb):
            F_k = Q_list[k] @ H                         # [ns_k, R]
            D_k = w_list[k]                             # [R]
            C_k = F_k * D_k                             # column-wise scale
            C_list.append(C_k)
            Yhat_k = C_k @ B.T                          # [ns_k, ng]
            Yhat_blocks.append(Yhat_k)
        S_stack = torch.vstack(Yhat_blocks)             # [S, ng]

        # ----- Y-step: solve Sylvester -----
        RHS = rho * (S_stack - U)
        Y = sylvester_cg(Ls, Lg, alpha=rho, gamma=gamma, RHS=RHS, Y0=Y, max_iter=300, tol=1e-6)

        # track ADMM primal residual for stopping
        prim_res = torch.linalg.norm(Y - S_stack) / (torch.linalg.norm(Y) + 1e-12)

        # ----- Q-step (Procrustes) -----
        Ytilde = Y + U
        for k in range(nb):
            rows = row_chunks[k]
            Ytilde_k = Ytilde[rows, :]                  # [ns_k, ng]
            W_k = H * w_list[k].unsqueeze(0) @ B.T      # H D_k B^T => [R, ng]
            Mk = X_list[k] @ W_k.T + (rho / 2.0) * (Ytilde_k @ W_k.T)   # [ns_k, R]
            Q_list[k] = _procrustes_Q(Mk)

        # ----- H-step -----
        SH = torch.zeros(rank, rank, device=device, dtype=dtype)
        RH = torch.zeros(rank, rank, device=device, dtype=dtype)
        for k in range(nb):
            D_k = w_list[k]                                   # [R]
            # G_k = diag(D_k) @ B.T  == (B * D_k).T
            G_k = (B * D_k).T                                 # [R, ng]
            SH += G_k @ G_k.T                                 # sum_k G_k G_k^T  (SPD)

            QtX  = Q_list[k].T @ X_list[k]                    # [R, ng]
            rows = row_chunks[k]
            QtYt = Q_list[k].T @ Ytilde[rows, :]              # [R, ng]
            RH  += (QtX + (rho / 2.0) * QtYt) @ G_k.T         # [R, R]

        # Solve (1 + rho/2) * H * SH = RH  →  H = solve(SH, RH.T/(1+rho/2))^T
        A = SH + epsR * eyeR                                  # tiny ridge keeps SPD
        H = solve_spd_then_fallback(A, (RH.T / (1.0 + rho / 2.0))).T # solve via Cholesky

        # ----- D-step -----
        BtB = B.T @ B                                   # [R, R]
        for k in range(nb):
            F_k = Q_list[k] @ H                         # [ns_k, R]
            FtF = F_k.T @ F_k                           # [R, R]
            Gram = (BtB * FtF)                          # Hadamard
            # rhs: diag(F_k^T X_k B) + (rho/2) diag(F_k^T Ytilde_k B)
            FtX_B = F_k.T @ X_list[k] @ B               # [R, R]
            rows = row_chunks[k]
            FtY_B = F_k.T @ Ytilde[rows, :] @ B         # [R, R]
            rhs_vec = _diag_of(FtX_B) + (rho / 2.0) * _diag_of(FtY_B)    # [R]
            A_D = (1.0 + rho / 2.0) * Gram + epsR * eyeR
            w_list[k] = solve_spd_then_fallback(A_D, rhs_vec.unsqueeze(1)).squeeze(1)

        # ----- B-step -----
        SB = torch.zeros(rank, rank, device=device, dtype=dtype)
        RB = torch.zeros(ng,   rank, device=device, dtype=dtype)
        for k in range(nb):
            rows = row_chunks[k]
            C_k = (Q_list[k] @ H) * w_list[k]                 # ns_k x R 
            SB += C_k.T @ C_k                                 # R x R
            RB += X_list[k].T @ C_k                           # ng x R
            RB += (rho / 2.0) * (Ytilde[rows, :].T @ C_k)     # ng x R

        # Solve (1 + rho/2) * SB * B^T = RB  →  B = solve(SB, RB^T/(1+rho/2))^T
        A = SB + epsR * eyeR                                   # tiny ridge keeps SPD
        B = solve_spd_then_fallback(A, (RB.T / (1.0 + rho/2.0))).T

        # column-rescale B and absorb into w_k 
        norms = torch.clamp(torch.linalg.norm(B, dim=0), min=1e-8)
        B /= norms
        for k in range(nb):
            w_list[k] = w_list[k] * norms

        # ----- Dual update -----
        # Rebuild S with updated factors
        Yhat_blocks = []
        for k in range(nb):
            F_k = Q_list[k] @ H
            C_k = F_k * w_list[k]
            Yhat_blocks.append(C_k @ B.T)
        S_stack = torch.vstack(Yhat_blocks)
        U = U + (Y - S_stack)

        # stopping
        if prim_res.item() < tol:
            break

    return Q_list, H, B, w_list, Y
