import torch.nn as nn
import torch
import torch.nn.functional as F
from model_base import BertTokenEmbeddingModel


class TokenSegmentTagDecoder(nn.Module):

    def __init__(self, char_emb, hidden_size, num_layers, dropout, out_size, out_dropout, num_labels):
        super(TokenSegmentTagDecoder, self).__init__()
        self.char_emb = char_emb
        self.encoder = nn.LSTM(input_size=char_emb.embedding_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=False,
                               batch_first=False,
                               dropout=dropout)
        self.decoder = nn.GRU(input_size=char_emb.embedding_dim,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              bidirectional=False,
                              batch_first=False,
                              dropout=dropout)
        self.out_dropout = nn.Dropout(out_dropout)
        self.char_out = nn.Linear(in_features=hidden_size, out_features=out_size)
        self.classifier = nn.Linear(in_features=hidden_size, out_features=num_labels)

    @property
    def enc_num_layers(self):
        return self.encoder.num_layers

    @property
    def dec_num_layers(self):
        return self.decoder.num_layers

    def forward(self, char_seq, enc_state, sos, eos, sep, max_len, max_num_tags, target_char_seq):
        mask = torch.ne(char_seq, 0)
        emb_chars = self.char_emb(char_seq[mask]).unsqueeze(1)
        enc_state = enc_state.view(1, 1, -1)
        enc_state = torch.split(enc_state, enc_state.shape[2] // self.enc_num_layers, dim=2)
        enc_state = torch.cat(enc_state, dim=0)
        enc_char_output, dec_char_state = self.encoder(emb_chars, enc_state)
        dec_char = sos
        dec_char_scores, dec_label_scores = [], []
        while len(dec_char_scores) < max_len and len(dec_label_scores) < max_num_tags:
            emb_dec_char = self.char_emb(dec_char).unsqueeze(1)
            dec_char_output, dec_char_state = self.decoder(emb_dec_char, dec_char_state)
            dec_char_output = self.out_dropout(dec_char_output)
            dec_char_output = self.char_out(dec_char_output)
            if target_char_seq is not None:
                dec_char = target_char_seq[len(dec_char_scores)].unsqueeze(0)
            else:
                dec_char = self.decode(dec_char_output).squeeze(0)
            dec_char_scores.append(dec_char_output)
            if torch.eq(dec_char, sep):
                dec_label_output = self.classifier(dec_char_output)
                dec_label_scores.append(dec_label_output)
            if torch.all(torch.eq(dec_char, eos)):
                if len(dec_label_scores) < max_num_tags:
                    dec_label_output = self.classifier(dec_char_output)
                    dec_label_scores.append(dec_label_output)
                break
        fill_len = max_len - len(dec_char_scores)
        dec_char_scores = torch.cat(dec_char_scores, dim=1)
        dec_label_scores = torch.cat(dec_label_scores, dim=1)
        return F.pad(dec_char_scores, (0, 0, 0, fill_len)), F.pad(dec_label_scores, (0, 0, 0, fill_len))

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=-1)


class MorphSegmentTagModel(nn.Module):

    def __init__(self, xtoken_emb: BertTokenEmbeddingModel, seg_tag_decoder: TokenSegmentTagDecoder):
        super(MorphSegmentTagModel, self).__init__()
        self.xtoken_emb = xtoken_emb
        self.seg_tag_decoder = seg_tag_decoder

    def forward(self, xtoken_seq, char_seq, special_symbols, num_tokens, max_form_len, max_num_tags, target_chars=None, target_tags=None):
        token_ctx = self.xtoken_emb(xtoken_seq)
        seg_scores = []
        tag_scores = []
        sos, eos, sep = special_symbols['<s>'], special_symbols['</s>'], special_symbols['<sep>']
        for cur_token_idx in range(num_tokens):
            cur_token_state = token_ctx[cur_token_idx + 1]
            cur_input_chars = char_seq[cur_token_idx]
            cur_target_chars = None
            if target_chars is not None:
                cur_target_chars = target_chars[cur_token_idx]
            cur_token_seg_scores, cur_token_tag_scores = self.seg_tag_decoder(cur_input_chars, cur_token_state,
                                                                              sos, eos, sep, max_form_len,
                                                                              max_num_tags, cur_target_chars)
            seg_scores.append(cur_token_seg_scores)
            tag_scores.append(cur_token_tag_scores)
        seg_scores = torch.cat(seg_scores, dim=0)
        tag_scores = torch.cat(tag_scores, dim=0)
        return seg_scores, tag_scores

    def decode(self, char_label_scores, tag_label_scores):
        return self.morph_emb.decode(char_label_scores), torch.argmax(tag_label_scores, dim=-1)
