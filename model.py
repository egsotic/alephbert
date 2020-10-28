import torch.nn as nn
# import torch.nn.functional as F
import torch


class TokenSeq2SeqMorphSeg(nn.Module):

    def __init__(self, char_emb, enc_hidden_size, enc_num_layers, enc_dropout, dec_num_layers, dec_dropout, out_size):
        super().__init__()
        self.char_emb = char_emb
        self.enc_input_size = self.char_emb.embedding_dim
        self.enc_hidden_size = enc_hidden_size
        self.enc_num_layers = enc_num_layers
        self.enc_dropout = enc_dropout
        self.dec_input_size = self.char_emb.embedding_dim
        self.dec_hidden_size = self.enc_hidden_size
        self.dec_num_layers = dec_num_layers
        self.dec_dropout = dec_dropout
        self.out_size = out_size
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

    def forward(self, in_token_char_seq, in_enc_state, out_char_seq, sos_char, eos_char,
                use_teacher_forcing):
        emb_chars = self.char_emb(in_token_char_seq).unsqueeze(1)
        enc_output, dec_state = self.encoder(emb_chars, in_enc_state)
        dec_char = sos_char
        dec_char_scores = []
        max_len = out_char_seq.shape[1]
        while len(dec_char_scores) < max_len:
            emb_dec_char = self.char_emb(dec_char).unsqueeze(1)
            dec_output, dec_state = self.decoder(emb_dec_char, dec_state)
            dec_output = self.out(dec_output)
            if use_teacher_forcing:
                dec_char = out_char_seq[:, len(dec_char_scores)]
            else:
                dec_char = self.decode(dec_output).squeeze(0)
            dec_char_scores.append(dec_output)
            if torch.all(torch.eq(dec_char, eos_char)):
                break
        fill_len = max_len - len(dec_char_scores)
        dec_char_scores.extend([dec_char_scores[-1]] * fill_len)
        return torch.cat(dec_char_scores, dim=1)

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=2)


class MorphSegModel(nn.Module):

    def __init__(self, xmodel, xtokenizer, char_emb, char_vocab, token_morph_seg):
        super().__init__()
        self.xmodel = xmodel
        self.xtokenizer = xtokenizer
        self.char_emb = char_emb
        self.char_vocab = char_vocab
        self.token_morph_seg = token_morph_seg
        self.loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.char_vocab['char2index']['<pad>'])

    def forward(self, xtokens, tokens, form_chars, max_tokens, max_form_len, sos_char, eos_char, use_teacher_forcing):
        dec_scores = []
        mask = xtokens != self.xtokenizer.pad_token_id
        token_ctx, sent_ctx = self.xmodel(xtokens, attention_mask=mask)
        cur_token_id = 1
        token_chars = tokens[tokens[:, :, 0] == cur_token_id]
        while token_chars.nelement() > 0:
            xtoken_ids = token_chars[:, 1]
            token_state = torch.mean(token_ctx[:, xtoken_ids], dim=1).unsqueeze(1)
            token_form_chars = form_chars[:, (cur_token_id-1)*max_form_len:cur_token_id*max_form_len]
            char_seq = token_chars[:, 2]
            dec_token_scores = self.token_morph_seg(char_seq, token_state, token_form_chars, sos_char, eos_char,
                                                    use_teacher_forcing)
            dec_scores.append(dec_token_scores)
            cur_token_id += 1
            token_chars = tokens[tokens[:, :, 0] == cur_token_id]
        num_tokens = cur_token_id - 1
        fill_len = max_tokens - num_tokens
        dec_scores.extend([dec_scores[-1]] * fill_len)
        return num_tokens, torch.cat(dec_scores, dim=1)
        # dec_scores.extend([dec_scores[-1]] * fill_len)
        # dec_scores = torch.cat(dec_scores, dim=1)
        # return num_tokens, F.pad(input=dec_scores, pad=(0, 0, 0, fill_len, 0, 0), mode='constant', value=0)

    def loss(self, dec_scores, gold_chars):
        return self.loss_fct(dec_scores[0], gold_chars[0])

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=2)
