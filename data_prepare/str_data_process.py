import os
import math
import torch
import argparse
import pandas as pd
import numpy as np
"""
    The following code is improved based on JmcPPI. Paper link:
    https://doi.org/10.48550/arXiv.2503.04650
"""

def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def p_match_feature(x, all_for_assign):
    x_p = np.zeros((len(x), 7))
    for j in range(len(x)):
        if x[j] == 'ALA':
            x_p[j] = all_for_assign[0, :]
        elif x[j] == 'CYS':
            x_p[j] = all_for_assign[1, :]
        elif x[j] == 'ASP':
            x_p[j] = all_for_assign[2, :]
        elif x[j] == 'GLU':
            x_p[j] = all_for_assign[3, :]
        elif x[j] == 'PHE':
            x_p[j] = all_for_assign[4, :]
        elif x[j] == 'GLY':
            x_p[j] = all_for_assign[5, :]
        elif x[j] == 'HIS':
            x_p[j] = all_for_assign[6, :]
        elif x[j] == 'ILE':
            x_p[j] = all_for_assign[7, :]
        elif x[j] == 'LYS':
            x_p[j] = all_for_assign[8, :]
        elif x[j] == 'LEU':
            x_p[j] = all_for_assign[9, :]
        elif x[j] == 'MET':
            x_p[j] = all_for_assign[10, :]
        elif x[j] == 'ASN':
            x_p[j] = all_for_assign[11, :]
        elif x[j] == 'PRO':
            x_p[j] = all_for_assign[12, :]
        elif x[j] == 'GLN':
            x_p[j] = all_for_assign[13, :]
        elif x[j] == 'ARG':
            x_p[j] = all_for_assign[14, :]
        elif x[j] == 'SER':
            x_p[j] = all_for_assign[15, :]
        elif x[j] == 'THR':
            x_p[j] = all_for_assign[16, :]
        elif x[j] == 'VAL':
            x_p[j] = all_for_assign[17, :]
        elif x[j] == 'TRP':
            x_p[j] = all_for_assign[18, :]
        elif x[j] == 'TYR':
            x_p[j] = all_for_assign[19, :]
    return x_p


def r_match_feature(x, all_for_assign,dataset):
    if dataset=="PRI":
        x_p = np.zeros((len(x), 5))
        for j in range(len(x)):
            if x[j] == 'DA':
                x_p[j] = all_for_assign[0, :]
            elif x[j] == 'DC':
                x_p[j] = all_for_assign[1, :]
            elif x[j] == 'DG':
                x_p[j] = all_for_assign[2, :]
            elif x[j] == 'DT':
                x_p[j] = all_for_assign[3, :]
            elif x[j] == 'DU':
                x_p[j] = all_for_assign[4, :]
    else:
        x_p = np.zeros((len(x), 5))
        for j in range(len(x)):
            if x[j] == 'DA':
                x_p[j] = all_for_assign[0, :]
            elif x[j] == 'DC':
                x_p[j] = all_for_assign[1, :]
            elif x[j] == 'DG':
                x_p[j] = all_for_assign[2, :]
            elif x[j] == 'DT':
                x_p[j] = all_for_assign[3, :]
            elif x[j] == 'DU':
                x_p[j] = all_for_assign[4, :]
    return x_p


def read_atoms(file, P_chain):
    p_chains = P_chain.split(',')
    p_residues = {}
    for line in file:
        line = line.strip()
        if line.startswith("HETATM") or line.startswith("ATOM"):
            chain = line[20:22].strip()
            residue_id = str(line[22:27].strip())
            residue_name = line[17:20].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            id = chain + residue_id
            if residue_name in ['TYR', 'TRP', 'VAL', 'THR', 'SER', 'ARG', 'GLN', 'PRO', 'ASN', 'MET', 'LEU', 'LYS',
                                'ILE', 'HIS',
                                'GLY', 'PHE', 'GLU', 'ASP', 'CYS', 'ALA']:
                if id not in p_residues:
                    p_residues[id] = {'coords': [], 'name': residue_name}
                p_residues[id]['coords'].append((x, y, z))
    p_atoms = []
    p_x = []
    p_num = []
    for residue_id, data in p_residues.items():
        coords = data['coords']
        avg_x = np.mean([c[0] for c in coords])
        avg_y = np.mean([c[1] for c in coords])
        avg_z = np.mean([c[2] for c in coords])
        p_atoms.append((avg_x, avg_y, avg_z))
        p_x.append(data['name'])
        p_num.append(residue_id)
    return p_atoms, p_x, p_num


def r_read_atoms(file, P_chain):
    p_chains = P_chain.split(',')
    p_residues = {}
    for line in file:
        line = line.strip()
        if line.startswith("HETATM") or line.startswith("ATOM"):
            chain = line[20:22].strip()
            residue_id = str(line[22:26].strip())
            residue_name = line[17:20].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            id = chain + residue_id
            last_char = residue_name[-1].upper()
            if last_char in ['A', 'C', 'G', 'T', 'U']:
                residue_name = 'D' + last_char
            if residue_name in ['DA', 'DC', 'DG', 'DT', 'DU']:
                if id not in p_residues:
                    p_residues[id] = {'coords': [], 'name': residue_name}
                p_residues[id]['coords'].append((x, y, z))
    p_atoms = []
    p_x = []
    p_num = []
    for residue_id, data in p_residues.items():
        coords = data['coords']
        avg_x = np.mean([c[0] for c in coords])
        avg_y = np.mean([c[1] for c in coords])
        avg_z = np.mean([c[2] for c in coords])
        p_atoms.append((avg_x, avg_y, avg_z))
        p_x.append(data['name'])
        p_num.append(residue_id)
    return p_atoms, p_x, p_num


def compute_contacts(atoms, threshold):
    contacts = []
    for i in range(len(atoms) - 2):
        for j in range(i + 2, len(atoms)):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i, j))
                contacts.append((j, i))
    return contacts


def knn(atoms, k=5):
    x = np.zeros((len(atoms), len(atoms)))
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            x[i, j] = dist(atoms[i], atoms[j])
    index = np.argsort(x, axis=-1)
    contacts = []
    for i in range(len(atoms)):
        num = 0
        for j in range(len(atoms)):
            if index[i, j] != i and index[i, j] != i - 1 and index[i, j] != i + 1:
                contacts.append((i, index[i, j]))
                num += 1
            if num == k:
                break
    return contacts


def pdb_to_cm(file1, file2, p_chain, r_chain, threshold):
    p_atoms, p_x, p_num = read_atoms(file1, p_chain)
    r_atoms, r_x, r_num = r_read_atoms(file2, r_chain)
    p_r_contacts = compute_contacts(p_atoms, threshold)
    p_k_contacts = knn(p_atoms)
    r_r_contacts = compute_contacts(r_atoms, threshold)
    r_k_contacts = knn(r_atoms)
    return p_r_contacts, r_r_contacts, p_k_contacts, r_k_contacts, p_x, r_x


def data_processing(dataset):
    # ==================== 可配置参数集中在此 ====================
    # 数据路径
    base_data_dir = "./data_prepare"  # 数据根目录（可根据需要修改）
    protein_pdb = os.path.join(base_data_dir, "protein_pdb")
    NA_pdb = os.path.join(base_data_dir, "NA_pdb")
    csv_file_path = os.path.join(base_data_dir, f"{dataset}.csv")

    # 特征文件路径
    all_assign1_path = "./protein.txt"
    # 根据数据集名称选择 all_assign2 文件
    if dataset == "PRI":
        all_assign2_path = "./RNA.txt"
    else:
        all_assign2_path = "./DNA.txt"

    # 距离阈值
    distance = 10

    # 输出目录
    output_dir = "./processed_data"
    # ==========================================================

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prot_files1 = os.listdir(protein_pdb)
    prot_files2 = os.listdir(NA_pdb)

    all_for_assign1 = np.loadtxt(all_assign1_path)
    all_for_assign2 = np.loadtxt(all_assign2_path)

    node_list = []
    r_edge_list = []
    k_edge_list = []

    pra344_df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

    for i in range(len(pra344_df)):
        line = pra344_df.iloc[i, 0]
        p_chain = pra344_df.iloc[i, 1]
        r_chain = pra344_df.iloc[i, 4]
        p_file_name = line +".pdb"
        r_file_name = line + ".pdb"

        if p_file_name in prot_files1 and r_file_name in prot_files2:
            p_r_contacts, r_r_contacts, p_k_contacts, r_k_contacts, p_x, r_x = pdb_to_cm(
                open(os.path.join(protein_pdb, p_file_name), "r"),
                open(os.path.join(NA_pdb, r_file_name), "r"),
                p_chain, r_chain, distance)

            p_x = p_match_feature(p_x, all_for_assign1)
            r_x = r_match_feature(r_x, all_for_assign2, dataset)

            node_list.append([p_x, r_x])
            r_edge_list.append([p_r_contacts, r_r_contacts])
            k_edge_list.append([p_k_contacts, r_k_contacts])
            print(i, p_file_name)

    np.save(os.path.join(output_dir, f"protein.rball.edges.{dataset}.npy"), np.array(r_edge_list, dtype=object))
    np.save(os.path.join(output_dir, f"protein.knn.edges.{dataset}.npy"), np.array(k_edge_list, dtype=object))
    torch.save(node_list, os.path.join(output_dir, f"protein.nodes.{dataset}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Implementation")
    parser.add_argument("--dataset", type=str, default="PDI", help="Dataset name, e.g., PDI, PRI")
    args = parser.parse_args()

    data_processing(args.dataset)