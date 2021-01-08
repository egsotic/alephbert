import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenCharMorphModel(nn.Module):

    def __init__(self, char_emb, hidden_size, num_layers, enc_dropout, dec_dropout, out_size, out_dropout):
        super(TokenCharMorphModel, self).__init__()
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

    def forward(self, char_seq, enc_state, sos, eos, max_decode_len, target_char_seq=None):
        emb_chars = self.char_emb(char_seq).unsqueeze(1)
        enc_state = enc_state.view(1, 1, -1)
        enc_state = torch.split(enc_state, enc_state.shape[2] // self.enc_num_layers, dim=2)
        enc_state = torch.cat(enc_state, dim=0)
        enc_output, dec_state = self.encoder(emb_chars, enc_state)
        dec_char = sos
        dec_scores = []
        while len(dec_scores) < max_decode_len:
            emb_dec_char = self.char_emb(dec_char).unsqueeze(1)
            dec_output, dec_state = self.decoder(emb_dec_char, dec_state)
            dec_output = self.dropout(dec_output)
            dec_output = self.out(dec_output)
            if target_char_seq is not None:
                dec_char = target_char_seq[len(dec_scores)].unsqueeze(0)
            else:
                dec_char = self.decode(dec_output).squeeze(0)
            dec_scores.append(dec_output)
            if torch.all(torch.eq(dec_char, eos)):
                break
        fill_len = max_decode_len - len(dec_scores)
        dec_scores = torch.cat(dec_scores, dim=1)
        return F.pad(dec_scores, (0, 0, 0, fill_len))

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=2)


class MorphSegModel(nn.Module):

    def __init__(self, xmodel, xtokenizer, token_char_morph_model):
        super().__init__()
        self.xmodel = xmodel
        self.xtokenizer = xtokenizer
        self.token_char_morph_model = token_char_morph_model

    def forward(self, in_xtokens, in_token_chars, special_symbols, max_form_len, target_token_form_chars=None):
        scores = []
        mask = torch.ne(in_xtokens[:, :, 1], self.xtokenizer.pad_token_id)
        xcontext = self.xmodel(in_xtokens[:, :, 1], attention_mask=mask)
        token_ctx = xcontext.last_hidden_state
        cur_token_idx = 1
        cur_token_chars = in_token_chars[in_token_chars[:, :, 0] == cur_token_idx]
        while cur_token_chars.nelement() > 0:
            cur_token_ctx = token_ctx[in_xtokens[:, :, 0] == cur_token_idx]
            token_state = torch.mean(cur_token_ctx, dim=0)
            # token_state = torch.sum(cur_token_ctx, dim=0)
            input_chars = cur_token_chars[:, 1]
            target_chars = None
            if target_token_form_chars is not None:
                target_chars = target_token_form_chars[0, cur_token_idx-1, :, 1]
            token_scores = self.token_char_morph_model(input_chars, token_state, special_symbols['<s>'],
                                                       special_symbols['</s>'], max_form_len, target_chars)
            scores.append(token_scores)
            cur_token_idx += 1
            cur_token_chars = in_token_chars[in_token_chars[:, :, 0] == cur_token_idx]
        num_tokens = cur_token_idx - 1
        return num_tokens, torch.cat(scores, dim=1)

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=2)
