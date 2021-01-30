import torch
import torch.nn as nn
import torch.nn.functional as F
from seg_model import MorphSegModel


class MorphTagger(nn.Module):

    def __init__(self, morph_emb: MorphSegModel, hidden_size, num_layers, dropout, out_size, out_dropout):
        super(MorphTagger, self).__init__()
        self.morph_emb = morph_emb
        self.input_size = self.morph_emb.embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.out_size = out_size
        self.out_dropput = out_dropout
        self.encoder = nn.LSTM(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               bidirectional=True,
                               batch_first=False,
                               dropout=self.dropout)
        self.tag_out = nn.Linear(in_features=self.hidden_size*2, out_features=self.out_size)
        self.dropout = nn.Dropout(self.out_dropput)

    def embed_xtokens(self, input_xtokens):
        return self.morph_emb.embed_xtokens(input_xtokens)

    def forward(self, input_token_context, input_token_chars, special_symbols, num_tokens, max_form_len, max_num_tags,  target_chars=None):
        sos = special_symbols['</s>']
        sep = special_symbols['<sep>']
        pad = special_symbols['<pad>']
        morph_scores, morph_states = self.morph_emb(input_token_context, input_token_chars, special_symbols, num_tokens, max_form_len, target_chars)
        if target_chars is not None:
            morph_chars = target_chars
        else:
            morph_chars = self.morph_emb.decode(morph_scores).squeeze(0)
        token_mask = torch.eq(morph_chars[:num_tokens], sos)
        morph_mask = torch.eq(morph_chars[:num_tokens], sep)
        mask = torch.bitwise_or(token_mask, morph_mask)
        morph_tag_states = morph_states[mask]
        enc_scores, _ = self.encoder(morph_tag_states.unsqueeze(dim=1))
        enc_scores = self.dropout(enc_scores)
        tag_scores = self.tag_out(enc_scores)

        idxs, vals = torch.unique_consecutive(token_mask, return_counts=True)
        morph_chars = torch.split_with_sizes(morph_chars[:num_tokens].view(-1), tuple(vals))
        tag_sizes = []
        for token_morph_chars in morph_chars:
            if torch.all(torch.eq(token_morph_chars, sos)):
                continue
            if torch.all(torch.eq(token_morph_chars, pad)):
                continue
            token_morph_chars = token_morph_chars[token_morph_chars != pad]
            morph_mask = torch.eq(token_morph_chars, sep)
            tag_sizes.append(torch.sum(morph_mask).item() + 1)
        tag_scores = torch.split_with_sizes(tag_scores.squeeze(dim=1), tag_sizes)
        tag_scores = nn.utils.rnn.pad_sequence(tag_scores, batch_first=True)
        fill_len = max_num_tags - tag_scores.shape[1]
        tag_scores = F.pad(tag_scores, (0, 0, 0, fill_len))
        return morph_scores, tag_scores

    def decode(self, morph_label_scores, tag_label_scores):
        return self.morph_emb.decode(morph_label_scores), torch.argmax(tag_label_scores, dim=-1)
