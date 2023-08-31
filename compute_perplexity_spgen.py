import torch
import sys
import pandas as pd
sys.path.append('..')
from src.baselines.datasets import SPGenDataset
from tqdm.auto import tqdm
import numpy as np
import argparse
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from src.baselines.transformer import Models
from src.baselines.progen.modeling_progen import ProGenForCausalLM
from tokenizers import Tokenizer

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap
url = "tcp://localhost:12355"
torch.distributed.init_process_group(backend="nccl", init_method = url, world_size=1, rank=0)

spgen_checkpoints = {
    75: '../../SPGen/remote_generation/signal_peptide/outputs/SIM75_550_12500_64_6_5_0.1_64_100_0.0001_-0.03_99_weightsonly.pt',
    90: '../../SPGen/remote_generation/signal_peptide/outputs/SIM90_550_12500_64_6_5_0.1_64_100_0.0001_-0.03_99_weightsonly.pt',
    95: '../../SPGen/remote_generation/signal_peptide/outputs/SIM95_550_12500_64_6_5_0.1_64_100_0.0001_-0.03_99_weightsonly.pt',
    99: '../../SPGen/remote_generation/signal_peptide/outputs/SIM99_550_12500_64_6_5_0.1_64_100_0.0001_-0.03_99_weightsonly.pt',
}

def load_spgen_model(checkpoint: int = 99):
    state_dict = torch.load(spgen_checkpoints[checkpoint])
    model = Models.Transformer(
        27,
        27,
        107,
        proj_share_weight=True,
        embs_share_weight=True,
        d_k=64,
        d_v=64,
        d_model=550,
        d_word_vec=550,
        d_inner_hid=1100,
        n_layers=6,
        n_head=5,
        dropout=0.1)

    model.load_state_dict(state_dict)
    model.eval()

    return model

# Get the alphabets and add in a padding character (' '), a stop character ('.'), and a start character ('$').
# with open('../data/ctable_copies/ctable_token_master.pkl', 'rb') as f:
#     ctable = pickle.load(f)
def get_perplexity_batch(transformer, src_seq, src_positions, tgt_seq, tgt_positions):
    '''Adapted from Translator()._epoch().'''
    ppls = []

    loss_fn = torch.nn.CrossEntropyLoss()

    pred = transformer((src_seq, src_positions), (tgt_seq, tgt_positions))

    # process each sample in batch
    for idx in range(len(src_seq)):
        loss = loss_fn(pred[idx].view(-1, 27), tgt_seq[idx,1:].view(-1))
        ppls.append(torch.exp(loss).item())

    return ppls


def predict_spgen(model, loader):
        
    all_aa_logits = []
    all_aa_targets = []

    with torch.no_grad():

        ppl = []
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):


            proteins, prot_positions, sps, sp_positions = batch
            proteins, prot_positions, sps, sp_positions = proteins.to(device), prot_positions.to(device), sps.to(device), sp_positions.to(device)

            aa_logits = model((proteins,prot_positions), (sps, sp_positions)) #sp_len -1, last pos trimmed

            ppls = get_perplexity_batch(model, proteins, prot_positions, sps, sp_positions)

            ppl.extend(ppls)

    return np.array(ppl)



def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument('checkpoint')
    parser.add_argument('out_dir')
    parser.add_argument('--data', type=str, default = '../data/fitness_data/grasso_prepared.csv')
    parser.add_argument('--checkpoint', type = int, default=99)
    args = parser.parse_args()




    model = load_spgen_model(args.checkpoint)
    model.to(device)
    os.makedirs(args.out_dir, exist_ok=True)

    test_set = SPGenDataset(args.data)
    

    loader = torch.utils.data.DataLoader(test_set, collate_fn = test_set.collate_fn, batch_size=500)
    perplexities = predict_spgen(model, loader)

    df = test_set.df
    df['perplexity'] = perplexities
    df.to_csv(os.path.join(args.out_dir, args.data.split('/')[-1]))




if __name__ == '__main__':
    main()