import numpy as np

# attach shared latent space coordinates to each adata object 
def attach_QHD_embeddings(
    adatas_in_order,
    Q_list,
    H,
    w_list,
    key="X_parafac2",
    also_store_shape=False,
    store_normed=True,
):
    H_np = H.detach().cpu().numpy()
    R = H_np.shape[0]
    for k, ad in enumerate(adatas_in_order):
        Qk = Q_list[k].detach().cpu().numpy()
        wk = w_list[k].detach().cpu().numpy().reshape(1, R)
        Fk = Qk @ H_np
        Ck = Fk * wk
        if Ck.shape[0] != ad.n_obs:
            raise ValueError(f"Slice {k}: n_obs ({ad.n_obs}) != ns_k ({Ck.shape[0]}).")
        ad.obsm[key] = Ck.astype(np.float32)
        if also_store_shape:
            ad.obsm[f"{key}_shape"] = Fk.astype(np.float32)
        if store_normed:
            denom = np.linalg.norm(Ck, axis=1, keepdims=True) + 1e-8
            ad.obsm[f"{key}_rownorm"] = (Ck / denom).astype(np.float32)
        ad.uns[f"{key}_meta"] = {"w": wk.ravel().astype(float).tolist(), "R": int(R)}