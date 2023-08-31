import torch
import sys
import pandas as pd
from src.models.seq2seq_default import Seq2SeqTransformer
from src.utils.dataset import PredictionSeq2SeqDataset
from src.utils.alphabets import AA_TO_ID, CODON_TO_ID, CODON_TO_AA, DummyTaxonomyMapping
from tqdm.auto import tqdm
import numpy as np
import argparse
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ID_TO_AA = {v: k for k, v in AA_TO_ID.items()}

def load_model(weights, all_taxonomy_levels=True, codons=False):
    state_dict = torch.load(weights, map_location='cpu')['module']
    state_dict['tok_emb.embedding.weight'] = state_dict['generator.weight']
    from pretraining.models.transformer_lora import lorafy_torch_transformer_checkpoint
    state_dict = lorafy_torch_transformer_checkpoint(state_dict)
    model = Seq2SeqTransformer(
        12,
        12,
        1024,
        16,
        aa_vocab_size = 66 if codons else 23,
        org_vocab_size = [11097+1, 4414+1, 1448+1, 596+1, 232+1, 112+1, 3+1, 3+1] if all_taxonomy_levels else 11097 +1,
        dim_feedforward = 2048,
        dropout = 0.1,
        pad_idx = 0,        
        )
    model.load_state_dict(state_dict)
    model.eval()

    return model



import torch
def make_aa_logits_from_codon_logits(codon_logits: torch.Tensor, start_codon: str = 'ATG') -> torch.Tensor:
    '''
    Converts codon logits to amino acid logits.
    Note that this does not need to handle start codons, as when 
    doing next token prediction we never predict the start codon,
    as the targets are shifted by one position.
    
    '''

    # we do the summing in probs and then convert back to logits
    aa_probs = torch.zeros(codon_logits.shape[0], 23, device=codon_logits.device)

    codon_probs = torch.softmax(codon_logits, dim=-1)

    aa_probs[:, AA_TO_ID['<pad>']] = codon_probs[:, CODON_TO_ID['<pad>']]
    aa_probs[:, AA_TO_ID['<eos>']] = codon_probs[:, CODON_TO_ID['<eos>']]

    # special case on codon at position 0: could be a non-standard start codon. Can't just map to M.
    # instead, check what the actual start codon in the seq was and map that prob to M also instead.
    # start_codon_prob = codon_probs[0, CODON_TO_ID[start_codon]]
    # aa_probs[0, AA_TO_ID['M']] += start_codon_prob
    # for codon, aa in CODON_TO_AA.items():
    #     if codon != start_codon:
    #         aa_probs[0, AA_TO_ID[aa]] += codon_probs[0, CODON_TO_ID[codon]]

    for codon, aa in CODON_TO_AA.items():
        aa_probs[:, AA_TO_ID[aa]] += codon_probs[:, CODON_TO_ID[codon]]
    
    aa_logits = torch.log(aa_probs)
    return aa_logits

def make_aa_logits(aa_logits):
    '''
    Process regular AA logits exactly the same as remapped ones.
    softmax, log. Should be the same as CrossEntropyLoss but better be sure.
    '''
    aa_probs = torch.softmax(aa_logits, dim=-1)
    aa_logits = torch.log(aa_probs)
    return aa_logits

def make_aa_labels_from_codon_labels(codon_labels: torch.Tensor) -> torch.Tensor:
    '''
    Converts codon labels to amino acid labels.
    Note that this does not need to handle start codons, as when 
    doing next token prediction we never predict the start codon,
    as the targets are shifted by one position.
    
    '''
    aa_labels = torch.zeros(codon_labels.shape[0], device=codon_labels.device, dtype=torch.long)
    aa_labels[codon_labels == CODON_TO_ID['<pad>']] = AA_TO_ID['<pad>']
    aa_labels[codon_labels == CODON_TO_ID['<eos>']] = AA_TO_ID['<eos>']

    for codon, aa in CODON_TO_AA.items():
        aa_labels[codon_labels == CODON_TO_ID[codon]] = AA_TO_ID[aa]
    return aa_labels

def predict(model, loader, no_org=False, all_taxonomy_levels=True, translate_codons=False):
        
    ppl = []

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):


            proteins, prot_masks, sps, sp_masks, org_level_targets = batch
            proteins, prot_masks, sps, sp_masks, org_level_targets = proteins.to(device), prot_masks.to(device), sps.to(device), sp_masks.to(device), org_level_targets.to(device)
            if no_org:
                orgs = None
            if all_taxonomy_levels:
                orgs = org_level_targets
            else:
                orgs = org_level_targets[:,0] # species_id for conditioning token

            proteins = proteins.transpose(1,0)
            sps = sps.transpose(1,0)

            # reindex for correct next token prediction.
            sps_input = sps[:-1,:]
            sps_tgt = sps[1:,:]

            aa_logits, hidden_states, hidden_state_mask = model(proteins, sps_input, orgs)

            # default:
            sp_loss = torch.nn.functional.cross_entropy(aa_logits.reshape(-1, model.aa_vocab_size), sps_tgt.reshape(-1), reduction='mean', ignore_index=0)
            print(np.exp(sp_loss.mean().item()))
            for i in range(aa_logits.shape[1]):

                if translate_codons:

                    l = torch.nn.functional.nll_loss(
                        make_aa_logits_from_codon_logits(aa_logits[:, i, :]).reshape(-1, 23),
                        make_aa_labels_from_codon_labels(sps_tgt[:, i]).reshape(-1),
                        reduction='mean',
                        ignore_index=0,
                        ).item()
                    # import ipdb; ipdb.set_trace()
                else:
                    l = torch.nn.functional.nll_loss(
                        make_aa_logits(aa_logits[:, i, :]).reshape(-1, model.aa_vocab_size), 
                        sps_tgt[:, i].reshape(-1), 
                        reduction='mean',
                        ignore_index=0,
                        ).item()

                ppl.append(np.exp(l))

                # all_aa_logits.append(aa_logits[:, idx, :].detach().cpu())
                # all_aa_targets.append(sps_tgt[:, idx].detach().cpu())

            print(np.mean(ppl))

    # ppl = []
    # if translate_codons:
    #     for idx in range(len(all_aa_logits)):
    #         translated_logits = make_aa_probs_from_codon_probs(all_aa_logits[idx])
    #         translated_targets = torch.zeros_like(all_aa_targets[idx])
    #         for i in range(len(all_aa_targets[idx])):
    #             translated_targets[i] = CODON_TO_AA[CODON_TO_ID[all_aa_targets[idx][i].item()]]

    #         l = torch.nn.functional.cross_entropy(translated_logits.reshape(-1, 23), translated_targets.reshape(-1), reduction='mean',ignore_index=0).item()
    #         ppl.append(np.exp(l))

    # else:
    #     for idx in range(len(all_aa_logits)):
    #         # import ipdb; ipdb.set_trace()
    #         l = torch.nn.functional.cross_entropy(all_aa_logits[idx].reshape(-1, model.aa_vocab_size), all_aa_targets[idx].reshape(-1), reduction='mean',ignore_index=0).item()
    #         ppl.append(np.exp(l))


    return np.array(ppl)






def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('out_file')
    parser.add_argument('--all_taxonomy_levels', action='store_true')
    parser.add_argument('--no_prot', action='store_true')
    parser.add_argument('--no_org', action='store_true')
    parser.add_argument('--codons', action='store_true')

    parser.add_argument('--data', type=str, default = 'data/fitness_data/grasso_prepared.csv')
    parser.add_argument('--taxonomy_dir', type=str, default = 'data/taxonomy_mappings')

    args = parser.parse_args()

    model = load_model(args.checkpoint, args.all_taxonomy_levels, args.codons)
    model.to(device)


    # 1. Test set
    levels_to_use = ['species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom', 'superkingdom']
    test_set = PredictionSeq2SeqDataset(args.data, taxonomy_dir=args.taxonomy_dir, use_codons=args.codons, levels_to_use=levels_to_use)
    
    if args.no_prot:
        test_set.proteins = np.array([['<pad>']] * len(test_set))

    if args.no_org:
        test_set.organisms = np.zeros_like(test_set.organisms, dtype=int)
        test_set.taxonomy_labels = DummyTaxonomyMapping()



    loader = torch.utils.data.DataLoader(test_set, collate_fn = test_set.collate_fn, batch_size=500)
    perplexities = predict(model, loader, no_org = args.no_org, all_taxonomy_levels = args.all_taxonomy_levels, translate_codons=args.codons)

    df = test_set.df
    df['perplexity'] = perplexities
    df.to_csv(args.out_file)



if __name__ == '__main__':
    main()