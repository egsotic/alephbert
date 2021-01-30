import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenCharSegmentDecoder(nn.Module):

    def __init__(self, char_emb, hidden_size, num_layers, dropout, out_size, out_dropout):
        super(TokenCharSegmentDecoder, self).__init__()
        self.char_emb = char_emb
        self.enc_input_size = self.char_emb.embedding_dim
        self.enc_hidden_size = hidden_size
        self.enc_num_layers = num_layers
        self.enc_dropout = dropout
        self.dec_input_size = self.char_emb.embedding_dim
        self.dec_hidden_size = hidden_size
        self.dec_num_layers = num_layers
        self.dec_dropout = dropout
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
        self.char_out = nn.Linear(in_features=self.dec_hidden_size, out_features=self.out_size)
        self.dropout = nn.Dropout(self.out_dropput)

    def forward(self, char_seq, enc_state, sos, eos, max_len, target_char_seq):
        mask = torch.ne(char_seq, 0)
        emb_chars = self.char_emb(char_seq[mask]).unsqueeze(1)
        enc_state = enc_state.view(1, 1, -1)
        enc_state = torch.split(enc_state, enc_state.shape[2] // self.enc_num_layers, dim=2)
        enc_state = torch.cat(enc_state, dim=0)
        enc_output, dec_state = self.encoder(emb_chars, enc_state)
        dec_char = sos
        dec_scores, dec_states = [], []
        while len(dec_scores) < max_len:
            emb_dec_char = self.char_emb(dec_char).unsqueeze(1)
            dec_output, dec_state = self.decoder(emb_dec_char, dec_state)
            dec_output = self.dropout(dec_output)
            dec_output = self.char_out(dec_output)
            if target_char_seq is not None:
                dec_char = target_char_seq[len(dec_scores)].unsqueeze(0)
            else:
                dec_char = self.decode(dec_output).squeeze(0)
            dec_scores.append(dec_output)
            dec_states.append(dec_state.view(1, -1, self.dec_num_layers * dec_state.shape[2]))
            if torch.all(torch.eq(dec_char, eos)):
                break
        fill_len = max_len - len(dec_scores)
        dec_scores = torch.cat(dec_scores, dim=1)
        dec_states = torch.cat(dec_states, dim=1)
        return F.pad(dec_scores, (0, 0, 0, fill_len)), F.pad(dec_states, (0, 0, 0, fill_len))

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=-1)


class MorphSegModel(nn.Module):

    def __init__(self, xmodel, segment_decoder):
        super(MorphSegModel, self).__init__()
        self.xmodel = xmodel
        self.segment_decoder = segment_decoder

    @property
    def embedding_dim(self):
        return self.xmodel.config.hidden_size

    def embed_xtokens(self, input_xtokens):
        mask = torch.ne(input_xtokens[:, :, 1], 0)
        # xoutput = self.xmodel(input_xtokens[mask][:, 1].unsqueeze(dim=0))
        xoutput = self.xmodel(input_xtokens[:, :, 1])
        emb_xtokens = xoutput.last_hidden_state
        emb_tokens = []
        for i in range(len(input_xtokens)):
            # # groupby token_id
            # mask = torch.ne(input_xtokens[i, :, 1], 0)
            idxs, vals = torch.unique_consecutive(input_xtokens[i, :, 0][mask[i]], return_counts=True)
            token_emb_xtokens = torch.split_with_sizes(emb_xtokens[i][mask[i]], tuple(vals))
            # token_xcontext = {k.item(): v for k, v in zip(idxs, [torch.mean(t, dim=0) for t in token_emb_xtokens])}
            emb_tokens.append(torch.stack([torch.mean(t, dim=0) for t in token_emb_xtokens], dim=0))
        return emb_tokens

    def forward(self, input_token_context, input_token_chars, special_symbols, num_tokens, max_form_len,
                target_chars=None):
        sos, eos = special_symbols['<s>'], special_symbols['</s>']
        scores, states = [], []
        for cur_token_idx in range(num_tokens):
            cur_token_state = input_token_context[cur_token_idx + 1]
            cur_input_chars = input_token_chars[cur_token_idx]
            cur_target_chars = None
            if target_chars is not None:
                cur_target_chars = target_chars[cur_token_idx]
            cur_token_scores, cur_token_states = self.segment_decoder(cur_input_chars, cur_token_state, sos, eos,
                                                                      max_form_len, cur_target_chars)
            scores.append(cur_token_scores)
            states.append(cur_token_states)
        return torch.cat(scores, dim=0), torch.cat(states, dim=0)

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=-1)
