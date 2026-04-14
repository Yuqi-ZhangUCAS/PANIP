import argparse
from src.dataloader import *
from src.strmodule import *
from src.module import *
import os
import random

def batch_data(data, batch_size):
    num_batches = (len(data) + batch_size - 1) // batch_size
    batches = [data[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
    return batches

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_interaction", type=str, default="./model/PANIP.pth")
    parser.add_argument("--model_affinity", type=str, default="./model/PANIP_af.pth")
    parser.add_argument("--h5_path", type=str, default="./data_prepare/example_feature.h5")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--output_csv", type=str, default="./results/example_prediction.csv")
    parser.add_argument("--seed", type=int, default=2023)

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=300)
    return parser.parse_args()

def main():
    args = get_args()
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    dataset = PANIPDatasetpre(h5_path=args.h5_path)
    all_samples = dataset.samples
    dataloader = batch_data(all_samples, args.batch_size)

    model_interaction = PANIPmodel(args,device).to(device)
    sd_i = torch.load(args.model_interaction, map_location=device)
    model_interaction.load_state_dict(sd_i, strict=False)
    model_interaction.eval()

    model_affinity = PANIPmodel(args,device).to(device)
    sd_a = torch.load(args.model_affinity, map_location=device)
    model_affinity.load_state_dict(sd_a, strict=False)
    model_affinity.eval()

    all_ids = []
    all_pred_classes = []
    all_pred_probs = []
    all_site_classes = []
    all_site_probs = []
    all_affinity=[]

    with torch.no_grad():
        for batch in dataloader:
            ids, pred_classes, pred_probs, site_classes, site_probs,_ = model_interaction(batch)
            _, _, _,_, _, affinity = model_affinity(batch)
            all_affinity.extend(affinity)
            all_ids.extend(ids)
            all_pred_classes.extend(np.array(pred_classes).flatten().tolist())
            all_pred_probs.extend(np.array(pred_probs).flatten().tolist())
            for sc_array, sp_array in zip(site_classes, site_probs):
                all_site_classes.append(','.join(map(str, sc_array)))
                all_site_probs.append(','.join(map(str, sp_array)))

    df = pd.DataFrame({
        'id': all_ids,
        'pred_class': all_pred_classes,
        'pred_prob': all_pred_probs,
        'site_class': all_site_classes,
        'site_prob': all_site_probs,
        "delta G":all_affinity
    })

    df.to_csv('./results/example_prediction.csv', index=False)

if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()