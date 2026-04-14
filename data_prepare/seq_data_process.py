import argparse
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from wandb.old.summary import h5py


class Dataset:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.sample = []
    def pre_data(self,df):
        samples = []
        pdb_dict = {}
        for _, row in df.iterrows():
            pdb_id = row["PDB"]
            protein_chains = row["Protein chains"]
            rna_chains = row["Nucleic Acid chains"]
            name = pdb_id+'_'+protein_chains+"_"+pdb_id+'_'+rna_chains
            protein_sequences = row["Protein sequences"].split(",")
            rna_sequences = row["Nucleic Acid sequences"].split(",")
            protein_seq_dict = {seq.split(":")[0]: seq.split(":")[1] for seq in protein_sequences}
            rna_seq_dict = {seq.split(":")[0]: seq.split(":")[1] for seq in rna_sequences}

            if name not in pdb_dict:
                pdb_dict[name] = {
                    "protein_chains": [],
                    "rna_chains": [],
                    "protein_seqs": [],
                    "rna_seqs": [],
                }
            pdb_dict[name]["protein_chains"].append(protein_chains)
            pdb_dict[name]["rna_chains"].append(rna_chains)
            pdb_dict[name]["protein_seqs"].append(protein_seq_dict[protein_chains])
            pdb_dict[name]["rna_seqs"].append(rna_seq_dict[rna_chains].strip())

        for pdb_id, data in pdb_dict.items():
            samples.append({
                "pdb_id": pdb_id,
                "protein_chains": data["protein_chains"],
                "rna_chains": data["rna_chains"],
                "protein_seqs": data["protein_seqs"],
                "rna_seqs": data["rna_seqs"],
            })
        return  samples

    def get_data(self):
        self.sample=self.pre_data(self.data)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx]

class ProteinRNAFeatureExtractor:
    def __init__(self):
        self.esmc_model = ESMC.from_pretrained("esmc_300m").to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("D:/unirna_tf/weights/unirna_L16_E1024_DPRNA500M_STEP400K")
        self.unirna_model = AutoModel.from_pretrained("D:/unirna_tf/weights/unirna_L16_E1024_DPRNA500M_STEP400K")

    def extract_protein_features(self, protein_seq):
        esms = []
        for i in range(len(protein_seq)):
            protein = ESMProtein(sequence=protein_seq[i])
            with torch.no_grad():
                protein_tensor = self.esmc_model.encode(protein)
                logits_output = self.esmc_model.logits(
                    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
                )
                esm_embedding = logits_output.embeddings[:, 1:-1, :]
                esms.append(esm_embedding)
        esm_embeddings = torch.cat(esms, dim=1)# [1, L, 960]
        return esm_embeddings

    def extract_rna_features(self, rna_seqs):

        rnas = []
        for rna_seq in rna_seqs:
            inputs = self.tokenizer(rna_seq, return_tensors="pt")
            outputs = self.unirna_model(**inputs)
            rna_embedding = (outputs.last_hidden_state)[:, 1:-1, :]
            rnas.append(rna_embedding)
        rna_embeddings = torch.cat(rnas, dim=1)

        return rna_embeddings

def main(csv_path, out_path):

    dataset = Dataset(csv_path)
    dataset.get_data()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    feature_extractor = ProteinRNAFeatureExtractor()

    with h5py.File(out_path, 'w') as f:
        for batch_idx, batch in enumerate(dataloader):
            pdb_id = batch["pdb_id"][0]
            protein_seq = batch["protein_seqs"][0]
            rna_seq = batch["rna_seqs"][0]

            protein_features = feature_extractor.extract_protein_features(protein_seq)
            rna_features = feature_extractor.extract_rna_features(rna_seq)

            grp = f.create_group(f"sample_{batch_idx}")

            grp.create_dataset("pdb_id", data=pdb_id)
            grp.create_dataset("protein_features", data=protein_features.to(torch.float32).cpu().numpy())
            grp.create_dataset("rna_features", data=rna_features.detach().to(torch.float32).cpu().numpy())

            print(f"id: {batch_idx},PDB: {pdb_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Implementation")
    parser.add_argument("--csv_path", type=str, default="example.csv")
    parser.add_argument("--out_h5_path", type=str, default="example_feature.h5")
    args = parser.parse_args()

    main(args.csv_path,args.out_h5_path)


