import torch.nn as nn
import torch.nn.functional as F
import torch


# class TokenCharEmbedding(nn.Module):
#
#     def __init__(self, token_emb, char_emb):
#         super(TokenCharEmbedding, self).__init__()
#         self.token_emb = token_emb
#         self.token_dropout = nn.Dropout(0.1)
#         self.char_emb = char_emb
#         self.char_lstm = nn.LSTM(input_size=self.char_emb.embedding_dim, hidden_size=50, batch_first=True)
#
#     def forward(self, token_chars, token_char_lengths):
#         batch_size = token_chars.shape[0]
#         token_seq_length = token_chars.shape[1]
#         char_seq_length = token_chars.shape[2]
#         token_seq = token_chars[:, :, 0, 0]
#         char_seq = token_chars[:, :, :, 1]
#         char_lengths = token_char_lengths[:, :, 1]
#         embed_chars = self.char_emb(char_seq)
#         char_inputs = embed_chars.view(batch_size * token_seq_length, char_seq_length, -1)
#         char_outputs, char_hidden_state = self.char_lstm(char_inputs)
#         char_outputs = char_outputs[torch.arange(char_outputs.shape[0]), char_lengths.view(-1) - 1]
#         char_outputs = char_outputs.view(batch_size, token_seq_length, -1)
#         embed_tokens = self.token_emb(token_seq)
#         embed_tokens = self.token_dropout(embed_tokens)
#         embed_tokens = torch.cat((embed_tokens, char_outputs), dim=2)
#         return embed_tokens
#
#     @property
#     def embedding_dim(self):
#         return self.token_emb.embedding_dim + self.char_lstm.hidden_size


class TokenCharEmbedding(nn.Module):

    def __init__(self, char_emb, token_emb_dim=0):
        super(TokenCharEmbedding, self).__init__()
        self.char_emb = char_emb
        self.input_dim = char_emb.embedding_dim + token_emb_dim
        self.char_rnn = nn.LSTM(input_size=self.input_dim, hidden_size=50, batch_first=True,
                                bidirectional=False, num_layers=1, dropout=0.0)

    def forward(self, chars, embedded_token=None):
        embed_chars = self.emb(chars.view(1, -1))
        if embedded_token:
            embed_chars = torch.cat((embed_chars, embedded_token), dim=0)
        char_outputs, char_hidden_state = self.rnn(embed_chars)
        return char_outputs

    @property
    def embedding_dim(self):
        return self.char_rnn.hidden_size


class Attention(nn.Module):

    def __init__(self, hidden_size, vocab_size, max_seq_len):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attn = nn.Linear(hidden_size, max_seq_len)
        self.attn_combine = nn.Linear(hidden_size, hidden_size)

    def forward(self, embedded, hidden, enc_outputs):
        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden), dim=1)), dim=1)
        attn_weights = attn_weights[:, :enc_outputs.shape[1]]
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_outputs)
        output = torch.cat((embedded.unsqueeze(1), attn_applied), 2)
        return self.attn_combine(output)


class Decoder(nn.Module):
    def __init__(self, emb, hidden_size, vocab_size):
        super().__init__()
        self.embedding = emb
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.gru = nn.GRU(input_size=emb.embedding_dim, hidden_size=hidden_size,
                          num_layers=1, batch_first=True, dropout=0.0, bidirectional=False)
        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, seg_input, hidden_state):
        output = self.embedding(seg_input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden_state = self.gru(output, hidden_state)
        output = self.out(output)
        return output, hidden_state


class AttnDecoder(nn.Module):
    def __init__(self, char_emb, input_size, hidden_size, vocab_size, attn):
        super().__init__()
        self.char_emb = char_emb
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, batch_first=True, dropout=0.0, bidirectional=False)
        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.attn = attn

    def forward(self, emb_token, char_input, hidden_state, enc_outputs):
        emb_char = self.char_emb(char_input)
        output = torch.cat([emb_token, emb_char], dim=1)
        output = F.relu(output)
        output = self.attn(output, hidden_state.squeeze(1), enc_outputs)
        output, hidden_state = self.gru(output, hidden_state)
        output = self.out(output)
        return output, hidden_state


class SegModel(nn.Module):

    def __init__(self, xtokenizer, xmodel, char_vocab, decoder):
        super().__init__()
        self.xtokenizer = xtokenizer
        self.xmodel = xmodel
        self.char_vocab = char_vocab
        self.decoder = decoder
        self.loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    def forward(self, input_xtokens, input_mask, token_chars, output_chars):
        enc_output, enc_hs = self.xmodel(input_xtokens, attention_mask=input_mask)
        dec_hs = enc_hs.unsqueeze(1)
        dec_scores = []
        dec_char = output_chars[:, 0]
        cur_token_id = 0
        cur_token_emb = enc_output[:, 0]
        while True:
            dec_output, dec_hs = self.decoder(cur_token_emb, dec_char, dec_hs, enc_output)
            if dec_char == self.char_vocab['<s>'] or dec_char == self.char_vocab['</s>']:
                cur_token_id += 1
                cur_token_char = token_chars[0, token_chars[0, :, 0] == cur_token_id]
                if cur_token_char.nelement() == 0:
                    break
                cur_xtoken_ids = cur_token_char[:, 1].unique()
                cur_enc_output = enc_output[:, cur_xtoken_ids - 1]
                cur_token_emb = torch.mean(cur_enc_output, dim=1)
            dec_scores.append(dec_output)
            dec_char = output_chars[:, len(dec_scores)]
        return torch.cat(dec_scores, dim=1)

    def loss(self, dec_scores, gold_chars, gold_mask):
        return self.loss_fct(dec_scores[0], gold_chars[gold_mask][1:])

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=2)


class Model2(nn.Module):

    def __init__(self, xmodel, token_char_emb, encoder, decoder, char_vocab):
        super().__init__()
        self.xmodel = xmodel
        self.token_char_emb = token_char_emb
        self.encoder = encoder
        self.decoder = decoder
        self.char_vocab = char_vocab
        self.loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    def forward(self, input_xtoken_ids, input_mask, output_char_ids):
        enc_output, enc_hs = self.xmodel(input_xtoken_ids, attention_mask=input_mask)

        dec_hs = enc_hs.unsqueeze(1)
        gold_idx = 0
        scores = []
        dec_output_char_ind = output_char_ids[0][gold_idx]
        while output_char_ids[0][gold_idx].item() != self.char_vocab['</s>']:
            dec_output, dec_hs = self.decoder(dec_output_char_ind, dec_hs, enc_output)
            gold_idx += 1
            dec_output_char_ind = output_char_ids[0][gold_idx]
            # gold_seg_ind = xform_ids[0][gold_idx]
            # gold_mask = xform_mask[0][gold_idx]
            # dec_seg_ind = gold_seg_ind if gold_mask else torch.tensor(self.xtokenizer.vocab['[SEP]'],
            #                                                           dtype=torch.long)
            if output_char_ids[0][gold_idx].item() != self.char_vocab['</s>']:
                scores.append(dec_output)
                # xforms = self.xtokenizer.ids_to_tokens[dec_seg_ind.item()]
                # xchar_ids = torch.tensor([self.xtokenizer.vocab[c] for c in xforms])
                # dec_emb_seg = self.emb(xchar_ids)
        return torch.cat(scores, dim=1)

    def loss(self, seg_scores, gold_segs, seg_mask: torch.Tensor):
        # xforms = [self.xtokenizer.ids_to_tokens[xform.item()] for xform in gold_segs.squeeze(0)[seg_mask.squeeze(0) != 0]]
        # xchar_ids = torch.tensor([self.xtokenizer.vocab[c] for xform in xforms for c in xform])
        return self.loss_fct(seg_scores.squeeze(0), gold_segs.squeeze(0)[seg_mask.squeeze(0) != 0])
        # return self.loss_fct(seg_scores.squeeze(0), xchar_ids)

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=2)
