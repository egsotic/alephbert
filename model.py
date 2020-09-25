import torch.nn as nn
import torch


class TokenCharEmbedding(nn.Module):

    def __init__(self, token_emb, char_emb):
        super(TokenCharEmbedding, self).__init__()
        self.token_emb = token_emb
        self.token_dropout = nn.Dropout(0.1)
        self.char_emb = char_emb
        self.char_lstm = nn.LSTM(input_size=self.char_emb.embedding_dim, hidden_size=50, batch_first=True)

    def forward(self, token_chars, token_char_lengths):
        batch_size = token_chars.shape[0]
        token_seq_length = token_chars.shape[1]
        char_seq_length = token_chars.shape[2]
        token_seq = token_chars[:, :, 0, 0]
        char_seq = token_chars[:, :, :, 1]
        char_lengths = token_char_lengths[:, :, 1]
        embed_chars = self.char_emb(char_seq)
        char_inputs = embed_chars.view(batch_size * token_seq_length, char_seq_length, -1)
        char_outputs, char_hidden_state = self.char_lstm(char_inputs)
        char_outputs = char_outputs[torch.arange(char_outputs.shape[0]), char_lengths.view(-1) - 1]
        char_outputs = char_outputs.view(batch_size, token_seq_length, -1)
        embed_tokens = self.token_emb(token_seq)
        embed_tokens = self.token_dropout(embed_tokens)
        embed_tokens = torch.cat((embed_tokens, char_outputs), dim=2)
        return embed_tokens

    @property
    def embedding_dim(self):
        return self.token_emb.embedding_dim + self.char_lstm.hidden_size


class MorphSegSeq2SeqModel(nn.Module):

    def __init__(self, emb: nn.Embedding, xtokenizer, xmodel):
        super().__init__()
        self.emb = emb
        self.xtokenizer = xtokenizer
        self.xmodel = xmodel
        # self.encoder = nn.Linear(in_features=xmodel.config.hidden_size, out_features=300)
        self.decoder = nn.GRU(input_size=self.emb.embedding_dim, hidden_size=xmodel.config.hidden_size,
                              num_layers=1, batch_first=True, dropout=0.0, bidirectional=False)
        self.output = nn.Linear(in_features=self.decoder.hidden_size, out_features=self.xmodel.config.vocab_size)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, xtoken_ids, xtoken_mask, xform_ids, xform_mask):
        enc_xtokens, enc_hs = self.xmodel(xtoken_ids, attention_mask=xtoken_mask)
        # dec_hidden_state = self.encoder(enc_hs).unsqueeze(1)
        dec_hs = enc_hs.unsqueeze(1)
        gold_idx = -1
        scores = []
        dec_seg_ind = torch.tensor(self.xtokenizer.vocab['[CLS]'])
        while dec_seg_ind.item() != self.xtokenizer.vocab['[SEP]']:
            dec_emb_seg = self.emb(dec_seg_ind).view(1, 1, self.emb.embedding_dim)
            # dec_emb_seg = self.emb(dec_seg_ind)
            dec_scores, dec_hs = self.decoder(dec_emb_seg, dec_hs)
            out_scores = self.output(dec_scores)
            # dec_seg_ind = self.decode(dec_scores)
            gold_idx += 1
            gold_seg_ind = xform_ids[0][gold_idx]
            gold_mask = xform_mask[0][gold_idx]
            dec_seg_ind = gold_seg_ind if gold_mask else torch.tensor(self.xtokenizer.vocab['[SEP]'])
            if dec_seg_ind.item() != self.xtokenizer.vocab['[SEP]']:
                scores.append(out_scores)
        return torch.cat(scores, dim=1)

    def loss(self, seg_scores, gold_segs, seg_mask):
        return self.loss_fct(seg_scores.squeeze(0), gold_segs.squeeze(0)[seg_mask.squeeze(0)])

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=2)
