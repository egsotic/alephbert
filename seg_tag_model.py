import torch
import torch.nn as nn
import torch.nn.functional as F
from seg_model import MorphSegModel


class MorphTagger(nn.Module):

    def __init__(self, morph_emb: MorphSegModel, hidden_size, num_layers, dropout, out_size, out_dropout):
        super(MorphTagger, self).__init__()
        self.morph_emb = morph_emb
        self.encoder = nn.LSTM(input_size=morph_emb.embedding_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=True,
                               batch_first=False,
                               dropout=dropout)
        self.tag_out = nn.Linear(in_features=self.encoder.hidden_size*2, out_features=out_size)
        self.out_dropout = nn.Dropout(out_dropout)

    def embed_xtokens(self, input_xtokens):
        return self.morph_emb.embed_xtokens(input_xtokens)

    def forward(self, input_token_context, input_token_chars, special_symbols, num_tokens, max_form_len, max_num_tags,  target_chars=None):
        sos = special_symbols['</s>']
        sep = special_symbols['<sep>']
        morph_scores, morph_states = self.morph_emb(input_token_context, input_token_chars, special_symbols, num_tokens, max_form_len, target_chars)
        if target_chars is not None:
            morph_chars = target_chars
        else:
            morph_chars = self.morph_emb.decode(morph_scores).squeeze(0)

        sos_mask = torch.eq(morph_chars[:num_tokens], sos)
        sos_mask[:, -1] = True
        sos_mask = torch.bitwise_and(torch.eq(torch.cumsum(sos_mask, dim=1), 1), sos_mask)

        sep_mask = torch.eq(morph_chars[:num_tokens], sep)
        sep_mask = torch.bitwise_and(torch.eq(torch.cumsum(sos_mask, dim=1), 0), sep_mask)

        tag_mask = torch.bitwise_or(sos_mask, sep_mask)
        tag_states = morph_states[tag_mask]
        enc_tag_scores, _ = self.encoder(tag_states.unsqueeze(dim=1))
        enc_tag_scores = self.out_dropout(enc_tag_scores)
        tag_scores = self.tag_out(enc_tag_scores)

        tag_sizes = torch.sum(tag_mask, dim=1)
        tag_scores = torch.split_with_sizes(tag_scores.squeeze(dim=1), tuple(tag_sizes))

        tag_scores = nn.utils.rnn.pad_sequence(tag_scores, batch_first=True)
        fill_len = max_num_tags - tag_scores.shape[1]
        tag_scores = F.pad(tag_scores, (0, 0, 0, fill_len))
        return morph_scores, tag_scores

    def decode(self, morph_label_scores, tag_label_scores):
        return self.morph_emb.decode(morph_label_scores), torch.argmax(tag_label_scores, dim=-1)
