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

    def __init__(self, char_emb):
        super(TokenCharEmbedding, self).__init__()
        self.char_emb = char_emb
        self.char_lstm = nn.LSTM(input_size=self.char_emb.embedding_dim, hidden_size=50, batch_first=True,
                                 bidirectional=False, num_layers=1, dropout=0.0)

    def forward(self, token_chars):
        embed_chars = self.char_emb(token_chars.view(1, -1))
        char_outputs, char_hidden_state = self.char_lstm(embed_chars)
        return char_outputs

    @property
    def embedding_dim(self):
        return self.char_lstm.hidden_size


class Attention(nn.Module):

    def __init__(self, hidden_size, vocab_size, max_seq_len):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attn = nn.Linear(hidden_size, max_seq_len)
        self.attn_combine = nn.Linear(hidden_size, hidden_size)

    def forward(self, embedded, hidden, enc_outputs):
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), dim=1)), dim=1)
        attn_weights = attn_weights[:, :enc_outputs.shape[1]]
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_outputs)
        output = torch.cat((embedded, attn_applied), 2)
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
    def __init__(self, emb, input_size, hidden_size, vocab_size, attn):
        super().__init__()
        self.embedding = emb
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, batch_first=True, dropout=0.0, bidirectional=False)
        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.attn = attn

    def forward(self, seg_input, hidden_state, enc_outputs):
        output = self.embedding(seg_input).view(1, 1, -1)
        output = F.relu(output)
        output = self.attn(output, hidden_state, enc_outputs)
        output, hidden_state = self.gru(output, hidden_state)
        output = self.out(output)
        return output, hidden_state


class MorphSegSeq2SeqModel(nn.Module):

    def __init__(self, xtokenizer, xmodel, char_vocab, decoder):
        super().__init__()
        self.xtokenizer = xtokenizer
        self.xmodel = xmodel
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
