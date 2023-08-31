import torch
import pandas as pd
from tokenizers import Tokenizer
import numpy as np

class ProGenDataset(torch.utils.data.Dataset):
    '''Use with ProGen baseline.'''
    def __init__(
        self, 
        dataset_csv: str,
        signal_peptide_key: str = 'signal_peptide_aa',
        mature_protein_key: str = 'mature_protein_aa',
        tokenizer_json='baselines/progen_tokenizer.json', 
        reverse: bool=False,
        ) -> None:

        df = pd.read_csv(dataset_csv, index_col=0)

        self.df = df


        self.sps = df[signal_peptide_key].astype(str).values
        self.proteins = df[mature_protein_key].astype(str).values

        self.tokenizer = Tokenizer.from_file(tokenizer_json)

        self.reverse = reverse


    def __len__(self):
        return len(self.sps)

    def __getitem__(self, idx):

        sp = self.sps[idx]
        prot = self.proteins[idx]
        seq = sp + prot
        seq = seq[:1024]
        if self.reverse:
            seq = seq[::-1]

        tokenized = self.tokenizer.encode(seq)

        seq = torch.from_numpy(np.array(tokenized.ids))
        return seq

    @staticmethod
    def collate_fn(batch):

        seqs = batch
        seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)

        return seqs


SPGEN_AA_TO_ID = {
 ' ': 0,
 '$': 1,
 '.': 2,
 'A': 3,
 'C': 4,
 'D': 5,
 'E': 6,
 'F': 7,
 'G': 8,
 'H': 9,
 'I': 10,
 'K': 11,
 'L': 12,
 'M': 13,
 'N': 14,
 'P': 15,
 'Q': 16,
 'R': 17,
 'S': 18,
 'T': 19,
 'U': 20,
 'V': 21,
 'W': 22,
 'X': 23,
 'Y': 24,
 'Z': 25,
}
SPGEN_AA_TO_ID['B'] = SPGEN_AA_TO_ID['X']
# max_len_in = 107 # max length of prot seq (105 aa) + 2 for tokens
# max_len_out = 72

class SPGenDataset(torch.utils.data.Dataset):
    '''Use with Wu transformer model baseline.'''
    def __init__(
        self, 
        dataset_csv: str,
        signal_peptide_key: str = 'signal_peptide_aa',
        mature_protein_key: str = 'mature_protein_aa',
        max_aas: int = 105,
        ) -> None:
        
        df = pd.read_csv(dataset_csv, index_col=0)
        self.df = df


        self.sps = df[signal_peptide_key].astype(str).values
        self.proteins = df[mature_protein_key].astype(str).values

        self.stop_token = SPGEN_AA_TO_ID['.']
        self.start_token = SPGEN_AA_TO_ID['$']

        self.max_aas = max_aas


    def __len__(self):
        return len(self.sps)

    def __getitem__(self, idx):

        sp = self.sps[idx]
        prot =  self.proteins[idx]

        #  truncate prot
        keep_len = min(self.max_aas, len(prot))
        prot = prot[:keep_len]

        sp = np.array([self.start_token] + [SPGEN_AA_TO_ID[x] for x in sp] + [self.stop_token])
        prot = np.array([self.start_token] + [SPGEN_AA_TO_ID[x] for x in prot] + [self.stop_token])

        sp = torch.from_numpy(sp)
        prot = torch.from_numpy(prot)

        # prot_mask = torch.zeros_like(prot)
        # sp_mask = torch.zeros_like(sp)

        #         inst_data = np.array([
            #     inst + [Constants.PAD] * (max_len - len(inst))
            #     for inst in insts])

            # inst_position = np.array([
            #     [pos_i+1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(inst)]
            #     for inst in inst_data])
        sp_positions = torch.from_numpy(np.array([i+1 for i in range(len(sp))]))
        prot_positions = torch.from_numpy(np.array([i+1 for i in range(len(prot))]))


        return sp, sp_positions, prot, prot_positions


    @staticmethod
    def collate_fn(batch):

        sps, sp_positions, proteins, prot_positions = zip(*batch)

        proteins = torch.nn.utils.rnn.pad_sequence(proteins, batch_first=True, padding_value=0)
        sps = torch.nn.utils.rnn.pad_sequence(sps, batch_first=True, padding_value=0)
        prot_positions = torch.nn.utils.rnn.pad_sequence(prot_positions, batch_first=True, padding_value=0)
        sp_positions = torch.nn.utils.rnn.pad_sequence(sp_positions, batch_first=True, padding_value=0)


        return proteins, prot_positions, sps, sp_positions

