import os
import json
from typing import List, Dict
import numpy as np

AA_TO_ID = {
 '<pad>': 0,
 'A': 1,
 'R': 2,
 'N': 3,
 'D': 4,
 'C': 5,
 'Q': 6,
 'E': 7,
 'G': 8,
 'H': 9,
 'I': 10,
 'L': 11,
 'K': 12,
 'M': 13,
 'F': 14,
 'P': 15,
 'S': 16,
 'T': 17,
 'W': 18,
 'Y': 19,
 'V': 20,
#  'U': 18, 
#  'O': 16,
#  'B': 23,
#  'X': 24,
#  'Z': 25,
 '<eos>': 21,
}

CODON_TO_ID = {
    '<pad>': 0,
    'AAA': 1,
    'AAC': 2,
    'AAG': 3,
    'AAT': 4,
    'ACA': 5,
    'ACC': 6,
    'ACG': 7,
    'ACT': 8,
    'AGA': 9,
    'AGC': 10,
    'AGG': 11,
    'AGT': 12,
    'ATA': 13,
    'ATC': 14,
    'ATG': 15,
    'ATT': 16,
    'CAA': 17,
    'CAC': 18,
    'CAG': 19,
    'CAT': 20,
    'CCA': 21,
    'CCC': 22,
    'CCG': 23,
    'CCT': 24,
    'CGA': 25,
    'CGC': 26,
    'CGG': 27,
    'CGT': 28,
    'CTA': 29,
    'CTC': 30,
    'CTG': 31,
    'CTT': 32,
    'GAA': 33,
    'GAC': 34,
    'GAG': 35,
    'GAT': 36,
    'GCA': 37,
    'GCC': 38,
    'GCG': 39,
    'GCT': 40,
    'GGA': 41,
    'GGC': 42,
    'GGG': 43,
    'GGT': 44,
    'GTA': 45,
    'GTC': 46,
    'GTG': 47,
    'GTT': 48,
    'TAA': 49,
    'TAC': 50,
    'TAG': 51,
    'TAT': 52,
    'TCA': 53,
    'TCC': 54,
    'TCG': 55,
    'TCT': 56,
    'TGA': 57,
    'TGC': 58,
    'TGG': 59,
    'TGT': 60,
    'TTA': 61,
    'TTC': 62,
    'TTG': 63,
    'TTT': 64,
    '<eos>': 65,
    }

# TODO QC github copilot
# codons to their amino acids
CODON_TO_AA = {
    'AAA': 'K',
    'AAC': 'N',
    'AAG': 'K',
    'AAT': 'N',
    'ACA': 'T',
    'ACC': 'T',
    'ACG': 'T',
    'ACT': 'T',
    'AGA': 'R',
    'AGC': 'S',
    'AGG': 'R',
    'AGT': 'S',
    'ATA': 'I',
    'ATC': 'I',
    'ATG': 'M',
    'ATT': 'I',
    'CAA': 'Q',
    'CAC': 'H',
    'CAG': 'Q',
    'CAT': 'H',
    'CCA': 'P',
    'CCC': 'P',
    'CCG': 'P',
    'CCT': 'P',
    'CGA': 'R',
    'CGC': 'R',
    'CGG': 'R',
    'CGT': 'R',
    'CTA': 'L',
    'CTC': 'L',
    'CTG': 'L',
    'CTT': 'L',
    'GAA': 'E',
    'GAC': 'D',
    'GAG': 'E',
    'GAT': 'D',
    'GCA': 'A',
    'GCC': 'A',
    'GCG': 'A',
    'GCT': 'A',
    'GGA': 'G',
    'GGC': 'G',
    'GGG': 'G',
    'GGT': 'G',
    'GTA': 'V',
    'GTC': 'V',
    'GTG': 'V',
    'GTT': 'V',
    'TAA': '<eos>',
    'TAC': 'Y',
    'TAG': '<eos>',
    'TAT': 'Y',
    'TCA': 'S',
    'TCC': 'S',
    'TCG': 'S',
    'TCT': 'S',
    'TGA': '<eos>',
    'TGC': 'C',
    'TGG': 'W',
    'TGT': 'C',
    'TTA': 'L',
    'TTC': 'F',
    'TTG': 'L',
    'TTT': 'F',
    }

import torch
def make_aa_probs_from_codon_probs(codon_probs: torch.Tensor, start_codon: str = 'ATG') -> torch.Tensor:
    '''Converts codon probabilities to amino acid probabilities.'''

    aa_probs = torch.zeros((codon_probs.shape[0], len(AA_TO_ID)))
    aa_probs[:, AA_TO_ID['<pad>']] = codon_probs[:, CODON_TO_ID['<pad>']]
    aa_probs[:, AA_TO_ID['<eos>']] = codon_probs[:, CODON_TO_ID['<eos>']]

    # special case on codon at position 0: could be a non-standard start codon. Can't just map to M.
    # instead, check what the actual start codon in the seq was and map that prob to M also instead.
    start_codon_prob = codon_probs[0, CODON_TO_ID[start_codon]]
    aa_probs[0, AA_TO_ID['M']] += start_codon_prob
    for codon, aa in CODON_TO_AA.items():
        if codon != start_codon:
            aa_probs[0, AA_TO_ID[aa]] += codon_probs[0, CODON_TO_ID[codon]]

    for codon, aa in CODON_TO_AA.items():
        aa_probs[1:, AA_TO_ID[aa]] += codon_probs[1:, CODON_TO_ID[codon]]
    return aa_probs



class TaxonomyMapping():
    '''A container class to map from species id to the class labels of all taxonomy levels. Wraps the json files.'''
    def __init__(self, taxonomy_mapping_dir: str = 'data/taxonomy_mappings') -> None:
        if not os.path.exists(taxonomy_mapping_dir):
            raise ValueError('Directory does not exist.')

        
        self.species_to_values: Dict[str, Dict[int,int]] = {}
        self.values_to_class_ints: Dict[str, Dict[int,int]] = {}

        # for each taxonomy level, build two mappings:
        # species_id -> level_taxon_id (uniprot numbers)
        # level_taxon_id -> class_id (class nums starting from 1 - 0 is reserved for "missing")
        for f in os.listdir(taxonomy_mapping_dir):
            src, _, tgt = f.removesuffix('.json').split('_')
            if src != 'species':
                raise NotImplementedError('Only accept species_to_X mappings at the moment.')
            
            d = json.load(open(os.path.join(taxonomy_mapping_dir, f), 'r'))
            d = {int(k):int(v) for k,v in d.items()}
            self.species_to_values[tgt] = d
            tgt_vals_unique = set(list(d.values()))
            # remove -1 from set if it exists, as those are always mapped to 0 and don't need an int assigned here
            tgt_vals_unique.discard(-1)
            self.values_to_class_ints[tgt] = {ID:INT+1 for INT, ID in enumerate(tgt_vals_unique)}


        self.values_to_class_ints['species'] =  {ID:INT+1 for INT, ID in enumerate(d.keys())}



    def get_taxonomy_labels(self, species_id: int, levels_to_return: List[str]) -> np.ndarray:
        '''Returns an array of the class labels of the requested taxonomy levels.'''
        outlist = []
        for lvl in levels_to_return:
            if lvl == 'species':
                outlist.append(self.values_to_class_ints['species'][species_id])
            else:
                taxon_id = self.species_to_values[lvl][species_id]
                if taxon_id == -1:
                    outlist.append(0)
                else:
                    outlist.append(self.values_to_class_ints[lvl][taxon_id])


        return np.array(outlist)


class DummyTaxonomyMapping():
    '''A container class to map from species id to the class labels of all taxonomy levels. Wraps the json files.'''
    def __init__(self, dummy_token = 0) -> None:
        self.dummy_token = 0


    def get_taxonomy_labels(self, species_id: int, levels_to_return: List[str]) -> np.ndarray:
        '''Returns an array of the class labels of the requested taxonomy levels.'''
        outlist = []
        for lvl in levels_to_return:
            outlist.append(self.dummy_token)

        return np.array(outlist)