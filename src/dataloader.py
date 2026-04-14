import os
import random
import torch.utils.checkpoint as checkpoint
import dgl
import numpy as np
import torch
import pandas as pd
import h5py

class ProteinDatasetDGL(torch.utils.data.Dataset):
    """
        The following code is improved based on JmcPPI. Paper link:
        https://doi.org/10.48550/arXiv.2503.04650
    """
    def __init__(self, prot_r_edge_path, prot_k_edge_path, prot_node_path, dataset):

        prot_r_edge = np.load(prot_r_edge_path, allow_pickle=True)
        prot_k_edge = np.load(prot_k_edge_path, allow_pickle=True)
        prot_node = torch.load(prot_node_path, weights_only=False)

        self.protein_graph_list = []
        self.rna_graph_list = []

        for i in range(len(prot_node)):
            for k in range(2):
                prot_seq = []
                for j in range(prot_node[i][k].shape[0] - 1):
                    prot_seq.append((j, j + 1))
                    prot_seq.append((j + 1, j))

                prot_g = dgl.heterograph({
                    ('amino_acid', 'SEQ', 'amino_acid'): prot_seq,
                    ('amino_acid', 'STR_KNN', 'amino_acid'): prot_k_edge[i][k],
                    ('amino_acid', 'STR_DIS', 'amino_acid'): prot_r_edge[i][k]
                })

                prot_g.ndata['x'] = torch.FloatTensor(prot_node[i][k])

                if k == 0:
                    self.protein_graph_list.append(prot_g)

                elif k == 1:
                    self.rna_graph_list.append(prot_g)

    def __len__(self):
        return min(len(self.protein_graph_list), len(self.rna_graph_list))

    def __getitem__(self, idx):

        protein_data = self.protein_graph_list[idx]
        rna_data = self.rna_graph_list[idx]
        return protein_data, rna_data

    def get_data(self):
        return self.protein_graph_list, self.rna_graph_list


class PANIPDatasetpre:
    def __init__(self, h5_path):
        self.samples=[]
        with h5py.File(h5_path, 'r') as f:
            for i in range(len(f.keys())):
                sample_key = f"sample_{i}"
                grp = f[sample_key]
                pdb_id = grp["pdb_id"][()].decode('utf-8')
                protein_features = torch.tensor(grp["protein_features"][()])
                rna_features = torch.tensor(grp["rna_features"][()])
                sample_dict = {
                    "pdb_id": pdb_id,
                    "protein_features": protein_features,
                    "rna_features": rna_features
                }
                self.samples.append(sample_dict)






