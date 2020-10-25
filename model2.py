import torch.nn as nn
import torch


class Model(nn.Module):

    def __init__(self, xmodel, xtokenizer, char_emb, char_vocab, enc_num_layers, enc_dropout, dec_num_layers, dec_dropout):
        super().__init__()
        self.xmodel = xmodel
        self.xtokenizer = xtokenizer
        self.char_emb = char_emb
        self.char_vocab = char_vocab
        self.pad_index = char_vocab['char2index']['<pad>']
        self.enc_input_size = self.char_emb.embedding_dim
        self.enc_hidden_size = self.xmodel.config.hidden_size
        self.enc_num_layers = enc_num_layers
        self.enc_dropout = enc_dropout
        self.dec_input_size = self.char_emb.embedding_dim
        self.dec_hidden_size = self.enc_hidden_size
        self.dec_num_layers = dec_num_layers
        self.dec_dropout = dec_dropout
        self.out_size = len(self.char_vocab['char2index'])
        self.encoder = nn.GRU(input_size=self.enc_input_size,
                              hidden_size=self.enc_hidden_size,
                              num_layers=self.enc_num_layers,
                              bidirectional=False,
                              batch_first=True,
                              dropout=self.enc_dropout)
        self.decoder = nn.GRU(input_size=self.dec_input_size,
                              hidden_size=self.dec_hidden_size,
                              num_layers=self.dec_num_layers,
                              bidirectional=False,
                              batch_first=True,
                              dropout=self.dec_dropout)
        self.out = nn.Linear(in_features=self.dec_hidden_size, out_features=self.out_size)
        self.loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.pad_index)

    def forward(self, in_xtokens, in_mask, in_sent_token_chars, out_chars, max_tokens, max_form_len, use_teacher_forcing):
        dec_scores = []
        token_ctx, sent_ctx = self.xmodel(in_xtokens, attention_mask=in_mask)
        cur_token_id = 1
        token_chars = in_sent_token_chars[in_sent_token_chars[:, :, 0] == cur_token_id]
        while token_chars.nelement() > 0:
            chars = token_chars[:, 2].unsqueeze(0)
            emb_chars = self.char_emb(chars)
            xtoken_ids = token_chars[:, 1]
            enc_state = torch.mean(token_ctx[:, xtoken_ids], dim=1).unsqueeze(1)
            enc_output, enc_state = self.encoder(emb_chars, enc_state)
            dec_state = enc_state
            dec_char = out_chars[:, 0]
            dec_token_scores = []
            while len(dec_token_scores) < max_form_len and torch.any(torch.ne(dec_char,
                                                                              self.char_vocab['char2index']['</s>'])):
                emb_dec_char = self.char_emb(dec_char).unsqueeze(1)
                dec_output, dec_state = self.decoder(emb_dec_char, dec_state)
                dec_output = self.out(dec_output)
                dec_token_scores.append(dec_output)
                gold_char = out_chars[:, len(dec_scores) + len(dec_token_scores)]
                if use_teacher_forcing:
                    dec_char = gold_char
                else:
                    dec_char = self.decode(dec_output).squeeze(0)
            pad_scores = dec_token_scores[-1]
            while len(dec_token_scores) < max_form_len:
                dec_token_scores.append(pad_scores)
            dec_scores.extend(dec_token_scores)
            cur_token_id += 1
            token_chars = in_sent_token_chars[in_sent_token_chars[:, :, 0] == cur_token_id]
        while cur_token_id <= max_tokens:
            pad_scores = dec_scores[-1]
            dec_token_scores = []
            while len(dec_token_scores) < max_form_len:
                dec_token_scores.append(pad_scores)
            dec_scores.extend(dec_token_scores)
            cur_token_id += 1
        return torch.cat(dec_scores, dim=1)

    def loss(self, dec_scores, gold_chars):
        return self.loss_fct(dec_scores[0], gold_chars[0, 1:])

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=2)
