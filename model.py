import torch.nn as nn
# import torch.nn.functional as F
import torch


class TokenSeq2SeqMorphSeg(nn.Module):

    def __init__(self, char_emb, hidden_size, num_layers, enc_dropout, dec_dropout, out_size, out_dropout):
        super().__init__()
        self.char_emb = char_emb
        self.enc_input_size = self.char_emb.embedding_dim
        self.enc_hidden_size = hidden_size
        self.enc_num_layers = num_layers
        self.enc_dropout = enc_dropout
        self.dec_input_size = self.char_emb.embedding_dim
        self.dec_hidden_size = hidden_size
        self.dec_num_layers = num_layers
        self.dec_dropout = dec_dropout
        self.out_size = out_size
        self.out_dropput = out_dropout
        self.encoder = nn.GRU(input_size=self.enc_input_size,
                              hidden_size=self.enc_hidden_size,
                              num_layers=self.enc_num_layers,
                              bidirectional=False,
                              batch_first=False,
                              dropout=self.enc_dropout)
        self.decoder = nn.GRU(input_size=self.dec_input_size,
                              hidden_size=self.dec_hidden_size,
                              num_layers=self.dec_num_layers,
                              bidirectional=False,
                              batch_first=False,
                              dropout=self.dec_dropout)
        self.out = nn.Linear(in_features=self.dec_hidden_size, out_features=self.out_size)
        self.dropout = nn.Dropout(self.out_dropput)

    def forward(self, in_token_char_seq, in_token_state, sos, eos, max_len, target_char_seq):
        emb_chars = self.char_emb(in_token_char_seq).unsqueeze(1)

        # hidden = [n layers * n directions, batch size, hidden dim]
        enc_state = torch.split(in_token_state, in_token_state.shape[2] // self.enc_num_layers, dim=2)
        enc_state = torch.cat(enc_state, dim=0)
        # enc_state = tuple(torch.split(hs, hs.shape[2] // self.enc_num_layers, dim=2) for hs in in_token_state)
        # enc_state = tuple(torch.cat(hs, dim=0) for hs in enc_state)

        enc_output, dec_state = self.encoder(emb_chars, enc_state)
        dec_char = sos
        dec_char_scores = []
        while len(dec_char_scores) < max_len:
            emb_dec_char = self.char_emb(dec_char).unsqueeze(1)
            dec_output, dec_state = self.decoder(emb_dec_char, dec_state)
            dec_output = self.dropout(dec_output)
            dec_output = self.out(dec_output)
            if target_char_seq is not None:
                dec_char = target_char_seq[:, len(dec_char_scores)]
            else:
                dec_char = self.decode(dec_output).squeeze(0)
            dec_char_scores.append(dec_output)
            if torch.all(torch.eq(dec_char, eos)):
                break
        fill_len = max_len - len(dec_char_scores)
        dec_char_scores.extend([dec_char_scores[-1]] * fill_len)
        return torch.cat(dec_char_scores, dim=1)

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=2)


class MorphSegModel(nn.Module):

    def __init__(self, xmodel, xtokenizer, char_emb, token_morph_seg):
        super().__init__()
        self.xmodel = xmodel
        self.xtokenizer = xtokenizer
        self.char_emb = char_emb
        self.token_morph_seg = token_morph_seg

    def forward(self, xtokens, tokens, sos, eos, max_form_len, target_form_chars=None):
        scores = []
        mask = xtokens != self.xtokenizer.pad_token_id
        token_ctx, sent_ctx = self.xmodel(xtokens, attention_mask=mask)
        cur_token_id = 1
        token_chars = tokens[tokens[:, :, 0] == cur_token_id]
        while token_chars.nelement() > 0:
            xtoken_ids = token_chars[:, 1]
            token_state = torch.mean(token_ctx[:, xtoken_ids], dim=1).unsqueeze(1)
            # token_state = (torch.mean(token_ctx[:, xtoken_ids], dim=1).unsqueeze(1), sent_ctx.unsqueeze(1))
            input_chars = token_chars[:, 2]
            target_chars = None
            if target_form_chars is not None:
                target_chars = target_form_chars[:, (cur_token_id - 1) * max_form_len:cur_token_id * max_form_len]
            morph_scores = self.token_morph_seg(input_chars, token_state, sos, eos, max_form_len, target_chars)
            scores.append(morph_scores)
            cur_token_id += 1
            token_chars = tokens[tokens[:, :, 0] == cur_token_id]
        num_tokens = cur_token_id - 1
        # fill_len = max_tokens - num_tokens
        # dec_scores.extend([dec_scores[-1]] * fill_len)
        return num_tokens, torch.cat(scores, dim=1)
        # dec_scores.extend([dec_scores[-1]] * fill_len)
        # dec_scores = torch.cat(dec_scores, dim=1)
        # return num_tokens, F.pad(input=dec_scores, pad=(0, 0, 0, fill_len, 0, 0), mode='constant', value=0)

    # def loss(self, dec_scores, gold_chars):
    #     return self.loss_fct(dec_scores[0], gold_chars[0])
    def loss_prepare(self, dec_scores, gold_chars):
        return dec_scores[0], gold_chars[0]

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=2)


