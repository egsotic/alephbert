import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class BertTokenEmbeddingModel(nn.Module):

    def __init__(self, bert: BertModel, bert_tokenizer: BertTokenizer):
        super(BertTokenEmbeddingModel, self).__init__()
        self.bert = bert
        self.bert_tokenizer = bert_tokenizer

    @property
    def embedding_dim(self):
        return self.bert.config.hidden_size

    def forward(self, token_seq):
        mask = torch.ne(token_seq[:, :, 1], self.bert_tokenizer.pad_token_id)
        # xoutput = self.bert(input_xtokens[mask][:, 1].unsqueeze(dim=0))
        bert_output = self.bert(token_seq[:, :, 1])
        bert_emb_tokens = bert_output.last_hidden_state
        emb_tokens = []
        for i in range(len(token_seq)):
            # # groupby token_id
            # mask = torch.ne(input_xtokens[i, :, 1], 0)
            idxs, vals = torch.unique_consecutive(token_seq[i, :, 0][mask[i]], return_counts=True)
            token_emb_xtoken_split = torch.split_with_sizes(bert_emb_tokens[i][mask[i]], tuple(vals))
            # token_xcontext = {k.item(): v for k, v in zip(idxs, [torch.mean(t, dim=0) for t in token_emb_xtokens])}
            emb_tokens.append(torch.stack([torch.mean(t, dim=0) for t in token_emb_xtoken_split], dim=0))
        return emb_tokens


class SegmentDecoder(nn.Module):

    def __init__(self, char_emb: nn.Embedding, hidden_size, num_layers, dropout, char_dropout, char_out_size,
                 num_labels: list):
        super(SegmentDecoder, self).__init__()
        self.char_emb = char_emb
        self.char_encoder = nn.GRU(input_size=char_emb.embedding_dim,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=False,
                                   batch_first=False,
                                   dropout=dropout)
        self.char_decoder = nn.GRU(input_size=char_emb.embedding_dim,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=False,
                                   batch_first=False,
                                   dropout=dropout)
        self.char_dropout = nn.Dropout(char_dropout)
        self.char_out = nn.Linear(in_features=self.decoder.hidden_size, out_features=char_out_size)
        self.classifiers = [nn.Linear(in_features=hidden_size, out_features=num) for num in num_labels]

    @property
    def enc_num_layers(self):
        return self.encoder.num_layers

    @property
    def dec_num_layers(self):
        return self.decoder.num_layers

    def forward(self, char_seq, enc_state, special_symbols, max_out_char_seq_len, target_char_seq, max_num_labels):
        char_scores, char_states, label_scores = [], [], []
        enc_output, dec_char_state = self.encode_(char_seq, enc_state)
        sos, eos, sep = special_symbols['<s>'], special_symbols['</s>'], special_symbols['<sep>']
        dec_char = sos
        for _ in self.classifiers:
            label_scores.append([])
        while (len(char_scores) < max_out_char_seq_len and
               (max_num_labels is None or all([len(scores) < max_num_labels for scores in label_scores]))):
            dec_output = self.decoder_step_(dec_char, dec_char_state, target_char_seq, len(char_scores), sep)
            dec_char, dec_char_output, dec_char_state, dec_labels_output = dec_output
            char_scores.append(dec_char_output)
            char_states.append(dec_char_state.view(1, -1, self.dec_num_layers * dec_char_state.shape[2]))
            if dec_labels_output:
                for output, scores in zip(dec_labels_output, label_scores):
                    scores.append(output)
            if torch.all(torch.eq(dec_char, eos)):
                break
        dec_labels_output = self.decoder_label_step_(char_scores[-1])
        if dec_labels_output:
            for output, scores in zip(dec_labels_output, label_scores):
                scores.append(output)
        char_fill_len = max_out_char_seq_len - len(char_scores)
        char_scores = torch.cat(char_scores, dim=1)
        char_scores_out = F.pad(char_scores, (0, 0, 0, char_fill_len))
        char_states = torch.cat(char_states, dim=1)
        char_states_out = F.pad(char_states, (0, 0, 0, char_fill_len))
        label_scores_out = []
        for scores in label_scores:
            label_fill_len = max_num_labels - len(scores)
            scores = torch.cat(scores, dim=1)
            label_scores_out.append(F.pad(scores, (0, 0, 0, label_fill_len)))
        return char_scores_out, char_states_out, label_scores_out

    def encode_(self, char_seq, enc_state):
        mask = torch.ne(char_seq, 0)
        emb_chars = self.char_emb(char_seq[mask]).unsqueeze(1)
        enc_state = enc_state.view(1, 1, -1)
        enc_state = torch.split(enc_state, enc_state.shape[2] // self.enc_num_layers, dim=2)
        enc_state = torch.cat(enc_state, dim=0)
        enc_output, enc_state = self.encoder(emb_chars, enc_state)
        return enc_output, enc_state

    def decoder_step_(self, cur_dec_char, dec_char_state, target_char_seq, num_scores, sep):
        emb_dec_char = self.char_emb(cur_dec_char).unsqueeze(1)
        dec_char_output, dec_char_state = self.decoder(emb_dec_char, dec_char_state)
        dec_char_output = self.out_dropout(dec_char_output)
        dec_char_output = self.char_out(dec_char_output)
        if target_char_seq is not None:
            next_dec_char = target_char_seq[num_scores].unsqueeze(0)
        else:
            next_dec_char = self.decode(dec_char_output).squeeze(0)
        if torch.eq(next_dec_char, sep):
            dec_labels_output = self.decoder_label_step_(dec_char_output)
        else:
            dec_labels_output = None
        return next_dec_char, dec_char_output, dec_char_state, dec_labels_output

    def decoder_label_step_(self, dec_char_output):
        return [classifier(dec_char_output) for classifier in self.classifiers]

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=-1)


class MorphSegmentModel(nn.Module):

    def __init__(self, xtoken_emb: BertTokenEmbeddingModel, segment_decoder: SegmentDecoder):
        super(MorphSegmentModel, self).__init__()
        self.xtoken_emb = xtoken_emb
        self.segment_decoder = segment_decoder

    @property
    def embedding_dim(self):
        return self.xtoken_emb.embedding_dim

    def forward(self, xtoken_seq, char_seq, special_symbols, num_tokens, max_form_len, max_num_labels,
                target_chars=None):
        token_ctx = self.xtoken_emb(xtoken_seq)
        out_char_scores, out_char_states = [], []
        out_label_scores = []
        for _ in self.segment_decoder.classifiers:
            out_label_scores.append([])
        for cur_token_idx in range(num_tokens):
            cur_token_state = token_ctx[cur_token_idx + 1]
            cur_input_chars = char_seq[cur_token_idx]
            cur_target_chars = None
            if target_chars is not None:
                cur_target_chars = target_chars[cur_token_idx]
            seg_output = self.segment_decoder(cur_input_chars, cur_token_state, special_symbols, max_form_len,
                                              cur_target_chars, max_num_labels)
            cur_token_segment_scores, cur_token_segment_states, cur_token_label_scores = seg_output
            out_char_scores.append(cur_token_segment_scores)
            out_char_states.append(cur_token_segment_states)
            for out_scores, seg_scores in zip(out_label_scores, cur_token_label_scores):
                out_scores.extend(seg_scores)
        out_char_scores = torch.cat(out_char_scores, dim=0)
        out_char_states = torch.cat(out_char_states, dim=0)
        out_label_scores = [torch.cat(label_scores, dim=0) for label_scores in out_label_scores]
        return out_char_scores, out_char_states, out_label_scores

    def decode(self, morph_seg_scores, label_scores):
        return self.morph_emb.decode(morph_seg_scores), [torch.argmax(scores, dim=-1) for scores in label_scores]


class MorphTagModel(nn.Module):

    def __init__(self, morph_emb: MorphSegmentModel, hidden_size, num_layers, dropout, out_size, out_dropout, crf=None):
        super(MorphTagModel, self).__init__()
        self.morph_emb = morph_emb
        self.encoder = nn.LSTM(input_size=morph_emb.embedding_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=True,
                               batch_first=False,
                               dropout=dropout)
        self.tag_out = nn.Linear(in_features=self.encoder.hidden_size*2, out_features=out_size)
        self.out_dropout = nn.Dropout(out_dropout)
        self.crf = crf

    def embed_xtokens(self, input_xtokens):
        return self.morph_emb.embed_xtokens(input_xtokens)

    def forward(self, xtoken_seq, char_seq, special_symbols, num_tokens, max_form_len, max_num_tags, target_chars=None):
        eos = special_symbols['</s>']
        sep = special_symbols['<sep>']
        morph_scores, morph_states = self.morph_emb(xtoken_seq, char_seq, special_symbols, num_tokens, max_form_len, target_chars)
        if target_chars is not None:
            morph_chars = target_chars
        else:
            morph_chars = self.morph_emb.decode(morph_scores).squeeze(0)

        eos_mask = torch.eq(morph_chars[:num_tokens], eos)
        eos_mask[:, -1] = True
        eos_mask = torch.bitwise_and(torch.eq(torch.cumsum(eos_mask, dim=1), 1), eos_mask)

        sep_mask = torch.eq(morph_chars[:num_tokens], sep)
        sep_mask = torch.bitwise_and(torch.eq(torch.cumsum(eos_mask, dim=1), 0), sep_mask)

        tag_mask = torch.bitwise_or(eos_mask, sep_mask)
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