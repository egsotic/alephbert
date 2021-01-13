import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenTagsDecoder(nn.Module):

    def __init__(self, tag_emb, hidden_size, num_layers, dropout, out_size, out_dropout):
        super(TokenTagsDecoder, self).__init__()
        self.tag_emb = tag_emb
        self.dec_input_size = self.tag_emb.embedding_dim
        self.dec_hidden_size = hidden_size
        self.dec_num_layers = num_layers
        self.dec_dropout = dropout
        self.out_size = out_size
        self.out_dropput = out_dropout
        self.decoder = nn.GRU(input_size=self.dec_input_size,
                              hidden_size=self.dec_hidden_size,
                              num_layers=self.dec_num_layers,
                              bidirectional=False,
                              batch_first=False,
                              dropout=self.dec_dropout)
        self.out = nn.Linear(in_features=self.dec_hidden_size, out_features=self.out_size)
        self.dropout = nn.Dropout(self.out_dropput)

    def forward(self, dec_state, sos, eos, max_decode_len, target_tag_seq):
        dec_state = dec_state.view(1, 1, -1)
        dec_state = torch.split(dec_state, dec_state.shape[2] // self.dec_num_layers, dim=2)
        dec_state = torch.cat(dec_state, dim=0)
        dec_tag = sos
        dec_scores = []
        while len(dec_scores) < max_decode_len:
            emb_dec_tag = self.tag_emb(dec_tag).unsqueeze(1)
            dec_output, dec_state = self.decoder(emb_dec_tag, dec_state)
            dec_output = self.dropout(dec_output)
            dec_output = self.out(dec_output)
            if target_tag_seq is not None:
                dec_tag = target_tag_seq[len(dec_scores)].unsqueeze(0)
            else:
                dec_tag = self.decode(dec_output).squeeze(0)
            dec_scores.append(dec_output)
            if torch.all(torch.eq(dec_tag, eos)):
                break
        fill_len = max_decode_len - len(dec_scores)
        dec_scores = torch.cat(dec_scores, dim=1)
        return F.pad(dec_scores, (0, 0, 0, fill_len))

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=2)


class TaggerModel(nn.Module):

    def __init__(self, xmodel, xtokenizer, token_decoder):
        super(TaggerModel, self).__init__()
        self.xmodel = xmodel
        self.xtokenizer = xtokenizer
        self.token_decoder = token_decoder

    def forward(self, input_xtokens, special_symbols, max_num_token_tags, target_token_tags=None):
        mask = torch.ne(input_xtokens[:, :, 1], self.xtokenizer.pad_token_id)
        xoutput = self.xmodel(input_xtokens[:, :, 1], attention_mask=mask)
        emb_xtokens = xoutput.last_hidden_state
        # groupby token_id
        xtoken_idxs, xtoken_vals = torch.unique(input_xtokens[:, :, 0][mask], return_counts=True)
        # token_xtokens = torch.split_with_sizes(input_xtokens[:, :, 1][mask], tuple(xtoken_vals))
        token_emb_xtokens = torch.split_with_sizes(emb_xtokens[mask], tuple(xtoken_vals))
        # token_xcontext = torch.stack([torch.mean(t, dim=0) for t in token_emb_xtokens], dim=0)
        token_xcontext = {k.item(): v for k, v in zip(xtoken_idxs[1:-1], [torch.mean(t, dim=0) for t in token_emb_xtokens[1:-1]])}
        sos, eos = special_symbols['<s>'], special_symbols['</s>']
        scores = []
        for cur_token_idx in token_xcontext:
            token_state = token_xcontext[cur_token_idx]
            target_tags = None
            if target_token_tags is not None:
                target_tags = target_token_tags[0, cur_token_idx - 1, :, 1]
            token_scores = self.token_decoder(token_state, sos, eos, max_num_token_tags, target_tags)
            scores.append(token_scores)
        return len(scores), torch.cat(scores, dim=1)

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=2)


class SegmentTaggerModel(nn.Module):

    def __init__(self):
        super(SegmentTaggerModel, self).__init__()

    def forward(self, input_xtokens, input_token_form_chars, target_token_tags=None):
        pass

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=2)
