from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from typing import List
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size, padding_idx=None):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


def generate_square_subsequent_mask(sz, device, dtype):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(dtype)


def create_mask(src, tgt, pad_idx, tgt_additional_tokens=0):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0] + tgt_additional_tokens

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, src.device, src.dtype).type(torch.bool)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=src.device).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class PhylogenyEmbedding(nn.Module):
    def __init__(self, level_sizes: List[int], embedding_size: int, padding_idx = 0):
        super(PhylogenyEmbedding, self).__init__()
        # [9583, 3570, 1117, 507, 208, 107, 4, 3]
        self.embedding_layers = nn.ModuleList([TokenEmbedding(x, embedding_size, padding_idx) for x in level_sizes])
        #levels_to_use = ['species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom', 'superkingdom']

    def forward(self, tokens):

        embeddings = []
        for idx, layer in enumerate(self.embedding_layers):
            embeddings.append(layer(tokens[:,idx]))

        return torch.stack(embeddings, axis=0).sum(axis=0)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 aa_vocab_size: int,
                 org_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 pad_idx: int = 0):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, aa_vocab_size)
        self.tok_emb = TokenEmbedding(aa_vocab_size, emb_size)
        self.generator.weight = self.tok_emb.embedding.weight
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

        if type(org_vocab_size) == list:
            self.organism_embedding = PhylogenyEmbedding(org_vocab_size, emb_size)
        else:
            self.organism_embedding = TokenEmbedding(org_vocab_size, emb_size)

        self.aa_vocab_size = aa_vocab_size
        self.org_vocab_size = org_vocab_size
        self.pad_idx = pad_idx

    def forward(self,
                src: Tensor,
                trg: Tensor,
                org: Tensor,
                ):

        src_emb = self.positional_encoding(self.tok_emb(src))
        tgt_emb = self.positional_encoding(self.tok_emb(trg))
        

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, trg, self.pad_idx, tgt_additional_tokens= 1 if org is not None else 0)
        memory_key_padding_mask = src_padding_mask


        if len(self.transformer.encoder.layers) >0:
            hidden_state_enc = self.transformer.encoder(src_emb, src_mask, src_padding_mask)
        else:
            hidden_state_enc = torch.zeros_like(tgt_emb[[0],:,:]) #vanilla pytorch decoder stack needs memory.
            memory_key_padding_mask = torch.zeros_like(tgt_padding_mask[:,[0]])
        

        
        

        if org is not None:
            # add conditioning token
            organism_embeddings = self.organism_embedding(org).unsqueeze(0) # (1, batch_size, embedding_dim)
            tgt_emb_with_org = torch.concat([organism_embeddings, tgt_emb], dim=0)
            tgt_padding_mask = torch.concat([tgt_padding_mask[:,[0]],tgt_padding_mask], dim=1) # batch_size, len
        else:
            tgt_emb_with_org = tgt_emb

        #hidden_state_enc_w_org = torch.concat([organism_embeddings, hidden_state_enc], dim=0)
        # also need to extend the memory padding mask.
        #memory_key_padding_mask = torch.concat([memory_key_padding_mask[:,[0]],memory_key_padding_mask], dim=1) # batch_size, len
        outs = self.transformer.decoder(tgt_emb_with_org, hidden_state_enc, tgt_mask, None, tgt_padding_mask, memory_key_padding_mask)
        # tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None

        if org is not None:
            outs = outs[1:] # reshift, org token only used internally.

        # outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
        #                         src_padding_mask, tgt_padding_mask, memory_key_padding_mask)

        return self.generator(outs), hidden_state_enc, src_padding_mask

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tok_emb(tgt)), memory,
                          tgt_mask)


    def generate_sequence(self, src, org, max_len = 70):
        '''Use simple greedy decoding to generate a target sequence.'''
        src_emb = self.positional_encoding(self.tok_emb(src))
        
        src_mask, _, src_padding_mask, _ = create_mask(src, src, self.pad_idx)
        memory_key_padding_mask = src_padding_mask


        hidden_state_enc = self.transformer.encoder(src_emb, src_mask, src_padding_mask)

        organism_embeddings = self.organism_embedding(org).unsqueeze(0) # (1, batch_size, embedding_dim)

        y = organism_embeddings
        for i in range(max_len-1):
            tgt_mask = generate_square_subsequent_mask(y.size(0), src.device)

            out = self.decode(y)
            out = out.transpose(0, 1)
            prob = self.generator(out[:, -1])
            _, next_token = torch.max(prob, dim=1)
            next_token = next_token.item()

            y = torch.cat([y, torch.ones(1, 1).type_as(src.data).fill_(next_token)], dim=0)

            if next_token == 26:
                break

        return y


