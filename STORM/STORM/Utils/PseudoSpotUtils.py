import numpy as np
import pandas as pd
import anndata
from tqdm import tqdm
import random

def generate_a_spot(
    sc_exp,
    min_cell_number_in_spot,
    max_cell_number_in_spot,
    max_cell_types_in_spot,
    generation_method,
):
    if generation_method == "cell":
        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)
        cell_list = list(sc_exp.obs.index.values)
        picked_cells = random.choices(cell_list, k=cell_num)
        return sc_exp[picked_cells]

    elif generation_method == "celltype":
        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)
        cell_type_num = random.randint(1, max_cell_types_in_spot)

        while True:
            cell_type_list_selected = random.choices(
                sc_exp.obs["cell_type"].value_counts().keys(), k=cell_type_num
            )
            if len(set(cell_type_list_selected)) == cell_type_num:
                break

        sc_exp_filter = sc_exp[sc_exp.obs["cell_type"].isin(cell_type_list_selected)]

        picked_cell_type = random.choices(cell_type_list_selected, k=cell_num)
        picked_cells = []
        for ct in picked_cell_type:
            data = sc_exp[sc_exp.obs["cell_type"] == ct]  # keep baseline behavior
            cell_list = list(data.obs.index.values)
            picked_cells.append(random.sample(cell_list, 1)[0])

        return sc_exp_filter[picked_cells]

    else:
        raise ValueError('generation_method should be "cell" or "celltype"')


def pseudo_spot_generation(
    sc_exp,
    idx_to_word_celltype,
    spot_num,
    min_cell_number_in_spot,
    max_cell_number_in_spot,
    max_cell_types_in_spot,
    generation_method,
):
    cell_type_num = len(sc_exp.obs["cell_type"].unique())
    generated_spots = []
    for _ in tqdm(range(spot_num), desc="Generating pseudo-spots (serial)"):
        one_spot = generate_a_spot(
            sc_exp,
            min_cell_number_in_spot,
            max_cell_number_in_spot,
            max_cell_types_in_spot,
            generation_method,
        )
        generated_spots.append(one_spot)

    pseudo_spots_table = np.zeros((spot_num, sc_exp.shape[1]), dtype=float)
    pseudo_fraction_table = np.zeros((spot_num, cell_type_num), dtype=float)

    for i, one_spot in enumerate(generated_spots):
        s = one_spot.X.sum(axis=0)
        pseudo_spots_table[i, :] = np.asarray(s).ravel()  
        type_idxs = one_spot.obs["cell_type_idx"].to_numpy()
        for tidx in type_idxs:
            pseudo_fraction_table[i, int(tidx)] += 1.0

    pseudo_spots_table = pd.DataFrame(pseudo_spots_table, columns=sc_exp.var.index.values)
    pseudo_spots = anndata.AnnData(X=pseudo_spots_table.values)
    pseudo_spots.obs.index = pseudo_spots_table.index 
    pseudo_spots.var.index = pseudo_spots_table.columns

    type_list = [idx_to_word_celltype[i] for i in range(cell_type_num)]
    pseudo_fraction_df = pd.DataFrame(pseudo_fraction_table, columns=type_list)
    pseudo_fraction_df["cell_num"] = pseudo_fraction_df.sum(axis=1)

    for ct in type_list:
        pseudo_fraction_df[ct] = pseudo_fraction_df[ct] / pseudo_fraction_df["cell_num"]

    pseudo_spots.obs = pseudo_spots.obs.join(pseudo_fraction_df)
    return pseudo_spots
