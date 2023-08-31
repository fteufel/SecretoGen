import torch
import numpy as np
import pandas as pd
from .alphabets import AA_TO_ID, TaxonomyMapping, CODON_TO_ID
import random


def string_to_codon_tokens(string):
    codons = [string[i:i+3] for i in range(0, len(string), 3)]
    codons = [CODON_TO_ID[x] for x in codons]
    return codons

class Seq2SeqDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_csv: str, split_csv: str, taxonomy_dir:str, use_codons: bool = False, keep_idx = [0], max_aas: int = 120, levels_to_use=['species','genus', 'family', 'class', 'phylum', 'kingdom', 'superkingdom']) -> None:

        df = pd.read_csv(dataset_csv, index_col=0)
        split_df = pd.read_csv(split_csv, index_col=0)
        df = df.join(split_df)
        df = df.loc[df['cluster'].isin(keep_idx)]

        self.df = df
        self.use_codons = use_codons

        self.levels_to_use = levels_to_use
        self.taxonomy_labels = TaxonomyMapping(taxonomy_dir)

        self.sps = df['signal_peptide_dna'].astype(str).values if use_codons else df['signal_peptide_aa'].astype(str).values
        self.proteins = df['mature_protein_dna'].astype(str).values if use_codons else df['mature_protein_aa'].astype(str).values
        self.organisms = df['species_taxon_id'].astype(int).values
        self.stop_token = CODON_TO_ID['<eos>'] if use_codons else AA_TO_ID['<eos>']

        self.max_aas = max_aas * 3 if use_codons else max_aas

    def __len__(self):
        return len(self.sps)

    def __getitem__(self, idx):

        sp = self.sps[idx]
        prot =  self.proteins[idx]
        org = self.organisms[idx]

        # randomly truncate prot
        keep_len = min(self.max_aas, len(prot))
        prot = prot[:keep_len]

        sp = string_to_codon_tokens(sp) if self.use_codons else [AA_TO_ID[x] for x in sp]
        prot = string_to_codon_tokens(prot) if self.use_codons else [AA_TO_ID[x] for x in prot]

        sp = np.array(sp + [self.stop_token])
        prot = np.array(prot + [self.stop_token])

        sp = torch.from_numpy(sp)
        prot = torch.from_numpy(prot)

        prot_mask = torch.zeros_like(prot)
        sp_mask = torch.zeros_like(sp)

        orgs = self.taxonomy_labels.get_taxonomy_labels(org, levels_to_return=self.levels_to_use)

        return sp, sp_mask, prot, prot_mask, orgs


    @staticmethod
    def collate_fn(batch):

        sps, sp_masks, proteins, prot_masks, orgs = zip(*batch)

        proteins = torch.nn.utils.rnn.pad_sequence(proteins, batch_first=True, padding_value=0)
        sps = torch.nn.utils.rnn.pad_sequence(sps, batch_first=True, padding_value=0)
        prot_masks = torch.nn.utils.rnn.pad_sequence(prot_masks, batch_first=True, padding_value=1)
        sp_masks = torch.nn.utils.rnn.pad_sequence(sp_masks, batch_first=True, padding_value=1)

        orgs = torch.from_numpy(np.array(orgs))


        return proteins, prot_masks, sps, sp_masks, orgs


# the prev. dataset is meant for handling training data.
class PredictionSeq2SeqDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        dataset_csv: str,
        taxonomy_dir:str, 
        use_codons: bool = False,
        signal_peptide_key: str = 'signal_peptide_aa',
        mature_protein_key: str = 'mature_protein_aa',
        species_taxon_id_key: str = 'species_taxon_id',
        max_aas: int = 120, 
        levels_to_use=['species','genus', 'family', 'class', 'phylum', 'kingdom', 'superkingdom']) -> None:

        df = pd.read_csv(dataset_csv, index_col=0)

        self.df = df
        self.use_codons = use_codons

        self.levels_to_use = levels_to_use
        self.taxonomy_labels = TaxonomyMapping(taxonomy_dir)

        self.sps = df[signal_peptide_key].astype(str).values
        self.proteins = df[mature_protein_key].astype(str).values
        self.organisms = df[species_taxon_id_key].astype(int).values
        self.stop_token = CODON_TO_ID['<eos>'] if use_codons else AA_TO_ID['<eos>']

        self.max_aas = max_aas * 3 if use_codons else max_aas

    def __len__(self):
        return len(self.sps)

    def __getitem__(self, idx):

        sp = self.sps[idx]
        prot =  self.proteins[idx]
        org = self.organisms[idx]

        # randomly truncate prot
        keep_len = min(self.max_aas, len(prot))
        prot = prot[:keep_len]

        sp = string_to_codon_tokens(sp) if self.use_codons else [AA_TO_ID[x] for x in sp]
        prot = string_to_codon_tokens(prot) if self.use_codons else [AA_TO_ID[x] for x in prot]

        sp = np.array(sp + [self.stop_token])
        prot = np.array(prot + [self.stop_token])

        sp = torch.from_numpy(sp)
        prot = torch.from_numpy(prot)

        prot_mask = torch.zeros_like(prot)
        sp_mask = torch.zeros_like(sp)

        orgs = self.taxonomy_labels.get_taxonomy_labels(org, levels_to_return=self.levels_to_use)

        return sp, sp_mask, prot, prot_mask, orgs


    @staticmethod
    def collate_fn(batch):

        sps, sp_masks, proteins, prot_masks, orgs = zip(*batch)

        proteins = torch.nn.utils.rnn.pad_sequence(proteins, batch_first=True, padding_value=0)
        sps = torch.nn.utils.rnn.pad_sequence(sps, batch_first=True, padding_value=0)
        prot_masks = torch.nn.utils.rnn.pad_sequence(prot_masks, batch_first=True, padding_value=1)
        sp_masks = torch.nn.utils.rnn.pad_sequence(sp_masks, batch_first=True, padding_value=1)

        orgs = torch.from_numpy(np.array(orgs))


        return proteins, prot_masks, sps, sp_masks, orgs