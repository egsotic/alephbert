import pandas as pd
import numpy as np
from pathlib import Path
import logging
from transformers import AutoTokenizer
from bclm import treebank as tb
import fasttext_emb as ft


def _process_token_data(raw_df, xtokenizer):
    df = raw_df[['sent_id', 'token_id', 'token']].drop_duplicates()
    data_rows = []
    for index, row in df.iterrows():
        for xt in xtokenizer.tokenize(row.token):
            data_rows.append([row.sent_id, row.token_id, row.token, xt])
    return pd.DataFrame(data_rows, columns=['sent_id', 'token_id', 'token', 'xtoken'])


def _process_token_char_data(token_df):
    data_rows = []
    gb = token_df.groupby(token_df.sent_id)
    for sent_id, df in gb:
        for xtoken_id, row in enumerate(df.itertuples()):
            xtoken_chars = row.xtoken[2:] if row.xtoken[:2] == '##' else row.xtoken
            for c in xtoken_chars:
                data_rows.append(list(row[1:]) + [xtoken_id + 1, c])
    return pd.DataFrame(data_rows, columns=list(token_df.columns) + ['xtoken_id', 'char'])


def _process_morph_data(raw_df):
    data_rows = []
    gb = raw_df.groupby(raw_df.sent_id)
    for sent_id, df in gb:
        for morph_id, row in enumerate(df.itertuples()):
            data_rows.append([row.sent_id, row.token_id, row.token, morph_id + 1] + list(row[4:8]))
    return pd.DataFrame(data_rows, columns=['sent_id', 'token_id', 'token', 'morph_id'] + list(raw_df.columns[3:7]))


def _process_morph_char_data(morph_df, field_name):
    data_rows = []
    gb = morph_df.groupby(morph_df.sent_id)
    for sent_id, df in gb:
        for morph_id, row in enumerate(df.iterrows()):
            for c in row[1][field_name]:
                data_rows.append(list(row[1]) + [c])
    return pd.DataFrame(data_rows, columns=list(morph_df.columns) + ['char'])


def _convert_xtokens_to_ids(df, xtokenizer):
    xtoken_ids = xtokenizer.convert_tokens_to_ids(df.xtoken)
    return [xtokenizer.cls_token_id] + xtoken_ids + [xtokenizer.sep_token_id]


def _convert_chars_to_ids(df, char_vocab):
    return [char_vocab[c] for c in df.char.tolist()]


def _to_token_data(samples, xtokenizer):
    max_len = max([len(sample.token_df.xtoken) + 2 for sample in samples])
    sent_ids_arr = np.zeros((len(samples), max_len), dtype=np.long)
    xtokens_arr = np.zeros((len(samples), max_len), dtype=np.object)
    lengths_arr = np.zeros((len(samples), max_len), dtype=np.long)
    for i, sample in enumerate(samples):
        sample_xtokens = [xtokenizer.cls_token] + sample.token_df.xtoken.tolist() + [xtokenizer.sep_token]
        sample_sent_id = [sample.token_df.sent_id.array[0] for _ in sample_xtokens]
        sample_length = [len(sample_xtokens)] * max_len
        sample_xtokens += [xtokenizer.pad_token] * (max_len - len(sample_xtokens))
        sample_sent_id += [sample_sent_id[0]] * (max_len - len(sample_sent_id))
        sent_ids_arr[i, :] = sample_sent_id
        xtokens_arr[i, :] = sample_xtokens
        lengths_arr[i:, :] = sample_length
    rows = zip(sent_ids_arr.flatten(), xtokens_arr.flatten(), lengths_arr.flatten())
    columns = ['sent_id', 'xtoken', 'length']
    return pd.DataFrame(rows, columns=columns)


def _to_token_char_data(samples, xtokenizer):
    max_len = max([len(sample.token_char_df.char) for sample in samples])
    sent_ids_arr = np.zeros((len(samples), max_len), dtype=np.long)
    tokens_arr = np.zeros((len(samples), max_len), dtype=np.object)
    token_ids_arr = np.zeros((len(samples), max_len), dtype=np.long)
    xtokens_arr = np.zeros((len(samples), max_len), dtype=np.object)
    xtoken_ids_arr = np.zeros((len(samples), max_len), dtype=np.long)
    chars_arr = np.zeros((len(samples), max_len), dtype=np.object)
    lengths_arr = np.zeros((len(samples), max_len), dtype=np.long)
    for i, sample in enumerate(samples):
        sample_chars = sample.token_char_df.char.tolist()
        sample_tokens = sample.token_char_df.token.tolist()
        sample_token_ids = sample.token_char_df.token_id.tolist()
        sample_xtokens = sample.token_char_df.xtoken.tolist()
        sample_xtoken_ids = sample.token_char_df.xtoken_id.tolist()
        sample_sent_id = [sample.token_char_df.sent_id.array[0] for _ in sample_chars]
        sample_length = [len(sample_chars)] * max_len
        sample_chars += ['<pad>'] * (max_len - len(sample_chars))
        sample_tokens += ['<pad>'] * (max_len - len(sample_tokens))
        sample_token_ids += [-1] * (max_len - len(sample_token_ids))
        sample_xtokens += [xtokenizer.pad_token] * (max_len - len(sample_xtokens))
        sample_xtoken_ids += [-1] * (max_len - len(sample_xtoken_ids))
        sample_sent_id += [sample_sent_id[0]] * (max_len - len(sample_sent_id))
        sent_ids_arr[i, :] = sample_sent_id
        tokens_arr[i, :] = sample_tokens
        token_ids_arr[i, :] = sample_token_ids
        xtokens_arr[i, :] = sample_xtokens
        xtoken_ids_arr[i, :] = sample_xtoken_ids
        chars_arr[i, :] = sample_chars
        lengths_arr[i, :] = sample_length
    rows = zip(sent_ids_arr.flatten(), token_ids_arr.flatten(), tokens_arr.flatten(), xtoken_ids_arr.flatten(),
               xtokens_arr.flatten(), chars_arr.flatten(), lengths_arr.flatten())
    columns = ['sent_id', 'token_id', 'token', 'xtoken_id', 'xtoken', 'char', 'length']
    return pd.DataFrame(rows, columns=columns)


def _to_morph_char_data(samples, morph_property):
    processed_samples = []
    for sample in samples:
        if morph_property == 'form':
            sent_id = sample.form_char_df.sent_id.array[0]
            sample_chars = sample.form_char_df.char.tolist()
            sample_tokens = sample.form_char_df.token.tolist()
            sample_token_ids = sample.form_char_df.token_id.tolist()
            sample_morph_ids = sample.form_char_df.morph_id.to_numpy()
            sample_morph_segs = sample.form_char_df.form.tolist()
        else:
            sent_id = sample.lemma_char_df.sent_id.array[0]
            sample_chars = sample.lemma_char_df.char.tolist()
            sample_tokens = sample.lemma_char_df.token.tolist()
            sample_token_ids = sample.lemma_char_df.token_id.tolist()
            sample_morph_ids = sample.lemma_char_df.morph_id.to_numpy()
            sample_morph_segs = sample.lemma_char_df.lemma.tolist()
        morph_pos = sample_morph_ids[:-1] != sample_morph_ids[1:]
        morph_pos = np.where(morph_pos)[0]
        sample_morph_ids = sample_morph_ids.tolist()

        for i, pos in enumerate(morph_pos):
            sample_chars.insert(pos + i + 1, '<sep>')
            sample_tokens.insert(pos + i + 1, sample_tokens[pos + i])
            sample_token_ids.insert(pos + i + 1, sample_token_ids[pos + i])
            sample_morph_ids.insert(pos + i + 1, sample_morph_ids[pos + i])
            sample_morph_segs.insert(pos + i + 1, sample_morph_segs[pos + i])
        sample_token_ids = np.array(sample_token_ids)
        token_pos = sample_token_ids[:-1] != sample_token_ids[1:]
        token_pos = np.where(token_pos)[0]
        sample_token_ids = sample_token_ids.tolist()
        for pos in token_pos:
            sample_chars[pos] = '</s>'
        sample_chars.append('</s>')
        sample_tokens.append('</s>')
        sample_token_ids.append(sample_token_ids[-1])
        sample_morph_ids.append(sample_morph_ids[-1])
        sample_morph_segs.append('</s>')
        sample_sent_id = [sent_id for _ in sample_chars]
        processed_sample = (sample_sent_id, sample_tokens, sample_token_ids, sample_morph_ids, sample_morph_segs,
                            sample_chars)
        processed_samples.append(processed_sample)
    max_len = max([len(sample[0]) for sample in processed_samples])
    sent_ids_arr = np.zeros((len(processed_samples), max_len), dtype=np.long)
    tokens_arr = np.zeros((len(processed_samples), max_len), dtype=np.object)
    token_ids_arr = np.zeros((len(processed_samples), max_len), dtype=np.long)
    morph_ids_arr = np.zeros((len(processed_samples), max_len), dtype=np.long)
    morph_segs_arr = np.zeros((len(processed_samples), max_len), dtype=np.object)
    chars_arr = np.zeros((len(processed_samples), max_len), dtype=np.object)
    lengths_arr = np.zeros((len(processed_samples), max_len), dtype=np.long)
    for i, processed_sample in enumerate(processed_samples):
        (sample_sent_id, sample_tokens, sample_token_ids, sample_morph_ids, sample_morph_segs,
         sample_chars) = processed_sample
        sample_length = [len(sample_chars)] * max_len
        sample_chars += ['<pad>'] * (max_len - len(sample_chars))
        sample_tokens += ['<pad>'] * (max_len - len(sample_tokens))
        sample_token_ids += [-1] * (max_len - len(sample_token_ids))
        sample_morph_ids += [-1] * (max_len - len(sample_morph_ids))
        sample_morph_segs += ['<pad>'] * (max_len - len(sample_morph_segs))
        sample_sent_id += [sample_sent_id[0]] * (max_len - len(sample_sent_id))
        sent_ids_arr[i, :] = sample_sent_id
        tokens_arr[i, :] = sample_tokens
        token_ids_arr[i, :] = sample_token_ids
        morph_ids_arr[i, :] = sample_morph_ids
        morph_segs_arr[i, :] = sample_morph_segs
        chars_arr[i, :] = sample_chars
        lengths_arr[i, :] = sample_length
    rows = zip(sent_ids_arr.flatten(), token_ids_arr.flatten(), tokens_arr.flatten(), morph_ids_arr.flatten(),
               morph_segs_arr.flatten(), chars_arr.flatten(), lengths_arr.flatten())
    columns = ['sent_id', 'token_id', 'token', 'morph_id', morph_property, 'char', 'length']
    return pd.DataFrame(rows, columns=columns)


def _to_morph_token_char_data(morph_char_data):
    morph_field_type = morph_char_data.columns.tolist()[4]
    gb = morph_char_data.groupby(['sent_id', 'token_id'])
    max_morph_len = max([len(group) for key, group in gb if key[1] != -1])
    max_num_tokens = max([key[1] for key, group in gb])
    sent_gb = morph_char_data.groupby(['sent_id'])
    max_len = max_num_tokens * max_morph_len
    sent_ids_arr = np.zeros((len(sent_gb), max_len), dtype=np.long)
    token_ids_arr = np.zeros((len(sent_gb), max_len), dtype=np.long)
    tokens_arr = np.zeros((len(sent_gb), max_len), dtype=np.object)
    morph_ids_arr = np.zeros((len(sent_gb), max_len), dtype=np.long)
    property_type_arr = np.zeros((len(sent_gb), max_len), dtype=np.object)
    chars_arr = np.zeros((len(sent_gb), max_len), dtype=np.object)
    lengths_arr = np.zeros((len(sent_gb), max_len), dtype=np.long)
    morph_length_arr = np.zeros((len(sent_gb), max_len), dtype=np.long)
    max_num_tokens_arr = np.zeros((len(sent_gb), max_len), dtype=np.long)
    max_morph_len_arr = np.zeros((len(sent_gb), max_len), dtype=np.long)
    for sent_id, sent_group in sent_gb:
        token_gb = sent_group.groupby('token_id')
        num_tokens = max([token_id for token_id, token_group in token_gb])
        for token_id, token_group in token_gb:
            if token_id == -1:
                continue
            token = token_group.token.array[0]
            morph_length = len(token_group)
            # token_rows = []
            token_length = num_tokens * max_morph_len

            token_group['length'] = token_length
            token_group[f'{morph_field_type}_length'] = morph_length
            token_group['max_num_tokens'] = max_num_tokens
            token_group[f'max_{morph_field_type}_len'] = max_morph_len
            # token_rows = token_group.to_list()
            fill_row = [sent_id, token_id, token, -1, '<pad>', '<pad>', token_length, morph_length, max_num_tokens,
                        max_morph_len]
            fill_len = max_morph_len - len(token_group)
            fill_rows = [fill_row] * fill_len
            # token_rows.extend(fill_rows)
            token_rows = token_group.to_numpy().tolist() + fill_rows
            from_pos = (token_id - 1) * len(token_rows)
            to_pos = token_id * len(token_rows)
            sent_ids_arr[sent_id - 1, from_pos:to_pos] = [row[0] for row in token_rows]
            token_ids_arr[sent_id - 1, from_pos:to_pos] = [row[1] for row in token_rows]
            tokens_arr[sent_id - 1, from_pos:to_pos] = [row[2] for row in token_rows]
            morph_ids_arr[sent_id - 1, from_pos:to_pos] = [row[3] for row in token_rows]
            property_type_arr[sent_id - 1, from_pos:to_pos] = [row[4] for row in token_rows]
            chars_arr[sent_id - 1, from_pos:to_pos] = [row[5] for row in token_rows]
            lengths_arr[sent_id - 1, from_pos:to_pos] = [row[6] for row in token_rows]
            morph_length_arr[sent_id - 1, from_pos:to_pos] = [row[7] for row in token_rows]
            max_num_tokens_arr[sent_id - 1, from_pos:to_pos] = [row[8] for row in token_rows]
            max_morph_len_arr[sent_id - 1, from_pos:to_pos] = [row[9] for row in token_rows]
        fill_row = [sent_id, -1, '<pad>', -1, '<pad>', '<pad>', -1, -1, max_num_tokens, max_morph_len]
        fill_len = (max_num_tokens - num_tokens) * max_morph_len
        fill_rows = [fill_row] * fill_len
        from_pos = num_tokens * max_morph_len
        sent_ids_arr[sent_id - 1, from_pos:] = [row[0] for row in fill_rows]
        token_ids_arr[sent_id - 1, from_pos:] = [row[1] for row in fill_rows]
        tokens_arr[sent_id - 1, from_pos:] = [row[2] for row in fill_rows]
        morph_ids_arr[sent_id - 1, from_pos:] = [row[3] for row in fill_rows]
        property_type_arr[sent_id - 1, from_pos:] = [row[4] for row in fill_rows]
        chars_arr[sent_id - 1, from_pos:] = [row[5] for row in fill_rows]
        lengths_arr[sent_id - 1, from_pos:] = [row[6] for row in fill_rows]
        morph_length_arr[sent_id - 1, from_pos:] = [row[7] for row in fill_rows]
        max_num_tokens_arr[sent_id - 1, from_pos:] = [row[8] for row in fill_rows]
        max_morph_len_arr[sent_id - 1, from_pos:] = [row[9] for row in fill_rows]
    rows = zip(sent_ids_arr.flatten(), token_ids_arr.flatten(), tokens_arr.flatten(), morph_ids_arr.flatten(),
               property_type_arr.flatten(), chars_arr.flatten(), lengths_arr.flatten(), morph_length_arr.flatten(),
               max_num_tokens_arr.flatten(), max_morph_len_arr.flatten())
    columns = list(morph_char_data.columns) + [f'{morph_field_type}_length', 'max_num_tokens',
                                               f'max_{morph_field_type}_len']
    return pd.DataFrame(rows, columns=columns)


def _to_morph_data(samples):
    max_len = max([len(sample.morph_df.morph_id) for sample in samples])
    sent_ids = np.zeros((len(samples), max_len), dtype=np.long)
    tokens = np.zeros((len(samples), max_len), dtype=np.object)
    token_ids = np.zeros((len(samples), max_len), dtype=np.long)
    morph_ids = np.zeros((len(samples), max_len), dtype=np.long)
    morph_forms = np.zeros((len(samples), max_len), dtype=np.object)
    morph_lemmas = np.zeros((len(samples), max_len), dtype=np.object)
    morph_tags = np.zeros((len(samples), max_len), dtype=np.object)
    morph_feats = np.zeros((len(samples), max_len), dtype=np.object)
    lengths = np.zeros((len(samples), max_len), dtype=np.long)
    for i, sample in enumerate(samples):
        sample_tokens = sample.morph_df.token.tolist()
        sample_token_ids = sample.morph_df.token_id.tolist()
        sample_morph_ids = sample.morph_df.morph_id.tolist()
        sample_morph_forms = sample.morph_df.form.tolist()
        sample_morph_lemmas = sample.morph_df.lemma.tolist()
        sample_morph_tags = sample.morph_df.tag.tolist()
        sample_morph_feats = sample.morph_df.feats.tolist()
        sample_sent_id = [sample.morph_df.sent_id.array[0] for _ in sample_tokens]
        sample_length = [len(sample_tokens)] * max_len
        sample_tokens += ['<pad>'] * (max_len - len(sample_tokens))
        sample_token_ids += [-1] * (max_len - len(sample_token_ids))
        sample_morph_ids += [-1] * (max_len - len(sample_morph_ids))
        sample_morph_forms += ['<pad>'] * (max_len - len(sample_morph_forms))
        sample_morph_lemmas += ['<pad>'] * (max_len - len(sample_morph_lemmas))
        sample_morph_tags += ['<pad>'] * (max_len - len(sample_morph_tags))
        sample_morph_feats += ['<pad>'] * (max_len - len(sample_morph_feats))
        sample_sent_id += [sample_sent_id[0]] * (max_len - len(sample_sent_id))
        sent_ids[i, :] = sample_sent_id
        tokens[i, :] = sample_tokens
        token_ids[i, :] = sample_token_ids
        morph_ids[i, :] = sample_morph_ids
        morph_forms[i, :] = sample_morph_forms
        morph_lemmas[i, :] = sample_morph_lemmas
        morph_tags[i, :] = sample_morph_tags
        morph_feats[i, :] = sample_morph_feats
        lengths[i, :] = sample_length
    rows = zip(sent_ids.flatten(), token_ids.flatten(), tokens.flatten(), morph_ids.flatten(),  morph_forms.flatten(),
               morph_lemmas.flatten(), morph_tags.flatten(), morph_feats.flatten(), lengths.flatten())
    columns = ['sent_id', 'token_id', 'token', 'morph_id', 'form', 'lemma', 'tag', 'feats', 'length']
    return pd.DataFrame(rows, columns=columns)


def _to_morph_token_data(morph_data):
    rows = []
    gb = morph_data.groupby(['sent_id', 'token_id'])
    max_form_len = max([len(group) for key, group in gb if key[1] != -1])
    max_num_tokens = max([key[1] for key, group in gb])
    sent_gb = morph_data.groupby(['sent_id'])
    for sent_id, sent_group in sent_gb:
        token_gb = sent_group.groupby('token_id')
        num_tokens = max([token_id for token_id, token_group in token_gb])
        sent_rows = []
        for token_id, token_group in token_gb:
            if token_id == -1:
                continue
            token = token_group.token.array[0]
            morph_length = len(token_group)
            token_rows = []
            token_length = num_tokens * max_form_len
            for token_row in token_group.iterrows():
                token_row[1]['length'] = token_length
                token_row[1]['morph_length'] = morph_length
                token_row[1]['max_num_tokens'] = max_num_tokens
                token_row[1]['max_morph_len'] = max_form_len
                token_rows.append(token_row[1].to_list())
            fill_row = [sent_id, token_id, token, -1, '<pad>', '<pad>', '<pad>', '<pad>', token_length, morph_length,
                        max_num_tokens, max_form_len]
            fill_len = max_form_len - len(token_group)
            fill_rows = [fill_row] * fill_len
            sent_rows.extend(token_rows)
            sent_rows.extend(fill_rows)
        fill_row = [sent_id, -1, '<pad>', -1, '<pad>', '<pad>', '<pad>', '<pad>', -1, -1, max_num_tokens, max_form_len]
        fill_len = max_form_len * (max_num_tokens - num_tokens)
        fill_rows = [fill_row] * fill_len
        sent_rows.extend(fill_rows)
        rows.extend(sent_rows)
    columns = list(morph_data.columns) + ['morph_length', 'max_num_tokens', 'max_morph_len']
    return pd.DataFrame(rows, columns=columns)


class DataSample:

    def __init__(self, raw_df, token_df, token_char_df, morph_df, form_char_df, lemma_char_df):
        raw_sent_id = raw_df.sent_id.array[0]
        token_sent_id = token_df.sent_id.array[0]
        token_char_sent_id = token_char_df.sent_id.array[0]
        morph_sent_id = morph_df.sent_id.array[0]
        form_char_sent_id = form_char_df.sent_id.array[0]
        lemma_char_sent_id = lemma_char_df.sent_id.array[0]
        assert raw_sent_id == token_sent_id
        assert raw_sent_id == token_char_sent_id
        assert raw_sent_id == morph_sent_id
        assert raw_sent_id == form_char_sent_id
        assert raw_sent_id == lemma_char_sent_id
        self.raw_df = raw_df
        self.token_df = token_df
        self.token_char_df = token_char_df
        self.morph_df = morph_df
        self.form_char_df = form_char_df
        self.lemma_char_df = lemma_char_df


class ProcessedDataSamples:

    def __init__(self, xtokenizer, char_vocab, raw_df, token_df, token_char_df, morph_df, morph_form_char_df, morph_lemma_char_df):
        self.xtokenizer = xtokenizer
        self.char_vocab = char_vocab
        self.raw_df = raw_df
        self.token_df = token_df
        self.morph_df = morph_df
        self.token_char_df = token_char_df
        self.morph_form_char_df = morph_form_char_df
        self.morph_lemma_char_df = morph_lemma_char_df

    def save(self, partition_name, root_path):
        token_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_token.csv'
        logging.info(f'saving {token_file}')
        self.token_df.to_csv(str(token_file))
        token_char_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_token_char.csv'
        logging.info(f'saving {token_char_file}')
        self.token_char_df.to_csv(str(token_char_file))
        morph_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_morph.csv'
        logging.info(f'saving {morph_file}')
        self.morph_df.to_csv(str(morph_file))
        morph_form_char_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_morph_form_char.csv'
        logging.info(f'saving {morph_form_char_file}')
        self.morph_form_char_df.to_csv(str(morph_form_char_file))
        morph_lemma_char_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_morph_lemma_char.csv'
        logging.info(f'saving {morph_lemma_char_file}')
        self.morph_lemma_char_df.to_csv(str(morph_lemma_char_file))

    def to_data(self):
        samples = list(self)
        # samples = self[:10]
        token_data = _to_token_data(samples, self.xtokenizer)
        token_char_data = _to_token_char_data(samples, self.xtokenizer)
        form_char_data = _to_morph_char_data(samples, 'form')
        form_token_char_data = _to_morph_token_char_data(form_char_data)
        lemma_char_data = _to_morph_char_data(samples, 'lemma')
        lemma_token_char_data = _to_morph_token_char_data(lemma_char_data)
        morph_data = _to_morph_data(samples)
        morph_token_data = _to_morph_token_data(morph_data)
        return (token_data, token_char_data, form_char_data, form_token_char_data, lemma_char_data,
                lemma_token_char_data, morph_data, morph_token_data)

    def __len__(self):
        raw_gb = self.raw_df.groupby(self.raw_df.sent_id)
        return len(raw_gb)

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            return [self[i + 1] for i in indices]
        sent_raw_df = self.raw_df[self.raw_df.sent_id == key]
        sent_token_df = self.token_df[self.token_df.sent_id == key]
        sent_token_char_df = self.token_char_df[self.token_char_df.sent_id == key]
        sent_morph_df = self.morph_df[self.morph_df.sent_id == key]
        sent_form_char_df = self.morph_form_char_df[self.morph_form_char_df.sent_id == key]
        sent_lemma_char_df = self.morph_lemma_char_df[self.morph_lemma_char_df.sent_id == key]
        return DataSample(sent_raw_df, sent_token_df, sent_token_char_df, sent_morph_df, sent_form_char_df,
                          sent_lemma_char_df)

    def __iter__(self):
        for sent_id in self.raw_df.sent_id.unique():
            yield self[sent_id]

    class Builder:

        def __init__(self, xtokenizer, char_vocab):
            self.xtokenizer = xtokenizer
            self.char_vocab = char_vocab

        def get_samples(self, raw_df, partition_name):
            logging.info(f'processing {partition_name} {type(self.xtokenizer).__name__} tokens')
            token_df = _process_token_data(raw_df, self.xtokenizer)
            logging.info(f'processing {partition_name} {type(self.xtokenizer).__name__} token chars')
            token_char_df = _process_token_char_data(token_df)
            logging.info(f'processing {partition_name} {type(self.xtokenizer).__name__} morphemes')
            morph_df = _process_morph_data(raw_df)
            logging.info(f'processing {partition_name} {type(self.xtokenizer).__name__} morpheme form chars')
            morph_form_char_df = _process_morph_char_data(morph_df, 'form')
            logging.info(f'processing {partition_name} {type(self.xtokenizer).__name__} morpheme form chars')
            morph_lemma_char_df = _process_morph_char_data(morph_df, 'lemma')
            return ProcessedDataSamples(self.xtokenizer, self.char_vocab, raw_df, token_df, token_char_df, morph_df,
                                        morph_form_char_df, morph_lemma_char_df)

        def load_samples(self, raw_df, partition_name, root_path):
            token_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_token.csv'
            logging.info(f'loading {token_file}')
            token_df = pd.read_csv(token_file, index_col=0)
            token_char_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_token_char.csv'
            logging.info(f'loading {token_char_file}')
            token_char_df = pd.read_csv(token_char_file, index_col=0)
            morph_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_morph.csv'
            logging.info(f'loading {morph_file}')
            morph_df = pd.read_csv(morph_file, index_col=0)
            form_char_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_morph_form_char.csv'
            logging.info(f'loading {form_char_file}')
            form_char_df = pd.read_csv(form_char_file, index_col=0)
            lemma_char_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_morph_lemma_char.csv'
            logging.info(f'loading {lemma_char_file}')
            lemma_char_df = pd.read_csv(lemma_char_file, index_col=0)
            return ProcessedDataSamples(self.xtokenizer, self.char_vocab, raw_df, token_df, token_char_df, morph_df,
                                        form_char_df, lemma_char_df)


def _save_emb(partition):
    tokens = set(token for part in partition for token in partition[part].token)
    forms = set(form for part in partition for form in partition[part].form)
    lemmas = set(lemma for part in partition for lemma in partition[part].lemma)
    chars = set(c.lower() for word in list(lemmas) + list(forms) + list(tokens) for c in word)
    tokens = ['<pad>', '<sep>', '<s>', '</s>'] + sorted(list(tokens))
    forms = ['<pad>', '<sep>', '<s>', '</s>'] + sorted(list(forms))
    lemmas = ['<pad>', '<sep>', '<s>', '</s>'] + sorted(list(lemmas))
    chars = ['<pad>', '<sep>', '<s>', '</s>'] + sorted(list(chars))
    token_vectors, token2index = ft.get_word_vectors('he', Path('/Users/Amit/dev/fastText/models/cc.he.300.bin'), tokens)
    form_vectors, form2index = ft.get_word_vectors('he', Path('/Users/Amit/dev/fastText/models/cc.he.300.bin'), forms)
    lemma_vectors, lemma2index = ft.get_word_vectors('he', Path('/Users/Amit/dev/fastText/models/cc.he.300.bin'), lemmas)
    char_vectors, char2index = ft.get_word_vectors('he', Path('/Users/Amit/dev/fastText/models/cc.he.300.bin'), chars)
    ft.save_word_vectors(Path('data/ft_token.vec.txt'), token_vectors, token2index)
    ft.save_word_vectors(Path('data/ft_form.vec.txt'), form_vectors, form2index)
    ft.save_word_vectors(Path('data/ft_lemma.vec.txt'), lemma_vectors, lemma2index)
    ft.save_word_vectors(Path('data/ft_char.vec.txt'), char_vectors, char2index)


def _save_samples(partition, xtokenizer, char_vocab, processed_path):
    data_samples_builder = ProcessedDataSamples.Builder(xtokenizer, char_vocab)
    for part in partition:
        logging.info(f'saving {part} samples')
        samples = data_samples_builder.get_samples(partition[part], part)
        samples.save(part, processed_path)


def _save_data(partition, xtokenizer, char_vocab, processed_path):
    data_samples_builder = ProcessedDataSamples.Builder(xtokenizer, char_vocab)
    for part in partition:
        logging.info(f'loading {part} samples')
        samples = data_samples_builder.load_samples(partition[part], part, processed_path)
        (token_data, token_char_data, form_char_data, form_token_char_data, lemma_char_data,
         lemma_token_char_data, morph_data, morph_token_data) = samples.to_data()
        token_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_token.csv'
        logging.info(f'saving {token_file}')
        token_data.to_csv(str(token_file))
        token_char_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_token_char.csv'
        logging.info(f'saving {token_char_file}')
        token_char_data.to_csv(str(token_char_file))
        form_char_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_form_char.csv'
        logging.info(f'saving {form_char_file}')
        form_char_data.to_csv(str(form_char_file))
        form_token_char_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_form_token_char.csv'
        logging.info(f'saving {form_token_char_file}')
        form_token_char_data.to_csv(str(form_token_char_file))
        lemma_char_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_lemma_char.csv'
        logging.info(f'saving {lemma_char_file}')
        lemma_char_data.to_csv(str(lemma_char_file))
        lemma_token_char_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_lemma_token_char.csv'
        logging.info(f'saving {lemma_token_char_file}')
        lemma_token_char_data.to_csv(str(lemma_token_char_file))
        morph_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_morph.csv'
        logging.info(f'saving {morph_file}')
        morph_data.to_csv(str(morph_file))
        morph_token_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_morph_token.csv'
        logging.info(f'saving {morph_token_file}')
        morph_token_data.to_csv(str(morph_token_file))


def _load_char_emb():
    char_vectors, char2index = ft.load_word_vectors(Path('data/ft_char.vec.txt'))
    index2char = {char2index[c]: c for c in char2index}
    char_vocab = {'char2index': char2index, 'index2char': index2char}
    return char_vectors, char_vocab


def load_vocab(partition, xtokenizer):
    char_vectors, char_vocab = _load_char_emb()
    tags, feats = set(), set()
    for part in partition:
        morph_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_morph.csv'
        morph_data = pd.read_csv(str(morph_file), index_col=0)
        tags |= set(morph_data.tag)
        feats |= set(morph_data.feats)
    tag2index = {t: i for i, t in enumerate(sorted(list(tags)))}
    index2tag = {v: k for k, v in tag2index.items()}
    tag_vocab = {'tag2index': tag2index, 'index2tag': index2tag}
    feats2index = {f: i for i, f in enumerate(sorted(list(feats)))}
    index2feats = {v: k for k, v in feats2index.items()}
    feats_vocab = {'feats2index': feats2index, 'index2feats': index2feats}
    return char_vectors, char_vocab, tag_vocab, feats_vocab


def load_data(partition, xtokenizer, char_vocab, tag_vocab, feats_vocab):
    data = {}
    for part in partition:
        token_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_token.csv'
        logging.info(f'loading {token_file}')
        token_data = pd.read_csv(str(token_file), index_col=0)
        token_data['xtoken'] = token_data['xtoken'].apply(lambda x: xtokenizer.convert_tokens_to_ids(x))
        token_groups = token_data.groupby('sent_id')
        token_data = np.stack([sent_df.to_numpy() for sent_id, sent_df in token_groups], axis=0)

        token_char_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_token_char.csv'
        logging.info(f'loading {token_char_file}')
        token_char_data = pd.read_csv(str(token_char_file), index_col=0)
        token_char_data['char'] = token_char_data['char'].apply(lambda x:
                                                                char_vocab['char2index'][x.lower()])
        token_char_data = token_char_data[['sent_id', 'token_id', 'xtoken_id', 'char', 'length']]
        token_char_groups = token_char_data.groupby('sent_id')
        token_char_data = np.stack([sent_df.to_numpy() for sent_id, sent_df in token_char_groups], axis=0)

        form_token_char_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_form_token_char.csv'
        logging.info(f'loading {form_token_char_file}')
        form_token_char_data = pd.read_csv(str(form_token_char_file), index_col=0)
        form_token_char_data['char'] = form_token_char_data['char'].apply(lambda x:
                                                                          char_vocab['char2index'][x.lower()])
        form_token_char_data = form_token_char_data[['sent_id', 'token_id', 'morph_id', 'char', 'length',
                                                     'form_length', 'max_num_tokens', 'max_form_len']]
        form_token_char_groups = form_token_char_data.groupby('sent_id')
        form_token_char_data = np.stack([sent_df.to_numpy() for sent_id, sent_df in form_token_char_groups], axis=0)

        lemma_token_char_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_lemma_token_char.csv'
        logging.info(f'loading {lemma_token_char_file}')
        lemma_token_char_data = pd.read_csv(str(lemma_token_char_file), index_col=0)
        lemma_token_char_data['char'] = lemma_token_char_data['char'].apply(lambda x:
                                                                            char_vocab['char2index'][x.lower()])
        lemma_token_char_data = lemma_token_char_data[['sent_id', 'token_id', 'morph_id', 'char', 'length',
                                                       'lemma_length', 'max_num_tokens', 'max_lemma_len']]
        lemma_token_char_groups = lemma_token_char_data.groupby('sent_id')
        lemma_token_char_data = np.stack([sent_df.to_numpy() for sent_id, sent_df in lemma_token_char_groups], axis=0)

        morph_token_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_morph_token.csv'
        logging.info(f'loading {morph_token_file}')
        morph_token_data = pd.read_csv(str(morph_token_file), index_col=0)
        morph_token_data['tag'] = morph_token_data['tag'].apply(lambda x: tag_vocab['tag2index'][x])
        morph_token_data['feats'] = morph_token_data['feats'].apply(lambda x: feats_vocab['feats2index'][x])
        morph_token_data = morph_token_data[['sent_id', 'token_id', 'morph_id', 'tag', 'feats',
                                             'length', 'morph_length', 'max_num_tokens', 'max_morph_len']]
        morph_token_groups = morph_token_data.groupby(['sent_id'])
        morph_token_data = np.stack([sent_df.to_numpy() for _, sent_df in morph_token_groups], axis=0)

        data[part] = token_data, token_char_data, form_token_char_data, lemma_token_char_data, morph_token_data
    return data


# form_chars:
# [1 x max_form_char_len x
#  [sent_id, token_id, morph_id, char, length, form_length, max_num_tokens, max_form_len]]
def _to_segmented_rows(token_chars, form_chars, lemma_chars, morphemes, char_vocab, tag_vocab, feats_vocab):
    sent_id = token_chars[0, 0, 0]
    from_token_id = 1
    to_token_id = np.max(token_chars[0, :, 1]) + 1
    sent_rows = []
    node_id = 0
    for token_id in range(from_token_id, to_token_id):
        tokens = token_chars[0][token_chars[0, :, 1] == token_id]
        token_forms = form_chars[0][form_chars[0, :, 1] == token_id]
        token_lemmas = lemma_chars[0][lemma_chars[0, :, 1] == token_id]
        token_morphemes = morphemes[0][morphemes[0, :, 1] == token_id]
        chars = tokens[:, 3]
        chars = [char_vocab['index2char'][c] for c in chars]
        token = ''.join(chars)
        from_morph_id = np.min(token_morphemes[:, 2][np.where(token_morphemes[:, 2] > 0)])
        to_morph_id = np.max(token_morphemes[:, 2]) + 1
        token_rows = []
        for morph_id in range(from_morph_id, to_morph_id):
            chars = token_forms[token_forms[:, 2] == morph_id, 3]
            chars = [char_vocab['index2char'][c] for c in chars
                     if c != char_vocab['char2index']['</s>'] and c != char_vocab['char2index']['<sep>']]
            form = ''.join(chars) if len(chars) > 0 else '_'
            chars = token_lemmas[token_lemmas[:, 2] == morph_id, 3]
            chars = [char_vocab['index2char'][c] for c in chars
                     if c != char_vocab['char2index']['</s>'] and c != char_vocab['char2index']['<sep>']]
            lemma = ''.join(chars) if len(chars) > 0 else '_'
            tag_id = token_morphemes[token_morphemes[:, 2] == morph_id, 3][0]
            tag = tag_vocab['index2tag'][tag_id]
            feat_id = token_morphemes[token_morphemes[:, 2] == morph_id, 4][0]
            feat = feats_vocab['index2feats'][feat_id]
            # ['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token', 'is_gold']
            token_rows.append((sent_id, node_id, node_id + 1, form, lemma, tag, feat, token_id, token, True))
            node_id += 1
        sent_rows.extend(token_rows)
    return sent_rows


def _pack_padded_chars(padded_input_chars, num_tokens, max_len, char_vocab):
    padded_input_chars = padded_input_chars[:, :num_tokens * max_len]
    is_eos = True
    eos_char = char_vocab['char2index']['</s>']
    packed_output_chars = []
    tokens = 0
    for i, c in enumerate(padded_input_chars[0]):
        c = c.item()
        if i % max_len == (max_len - 1):
            c = eos_char
        if i % max_len == 0:
            is_eos = False
        if not is_eos:
            packed_output_chars.append(c)
            if c == eos_char:
                tokens += 1
        if c == eos_char:
            is_eos = True
    return np.array([packed_output_chars], dtype=np.long)


def _to_token_chars(chars, char_vocab):
    eos_char = char_vocab['char2index']['</s>']
    token_chars_mask = np.eq(chars, eos_char)
    token_ids = np.cumsum(token_chars_mask, axis=1) + 1
    token_ids -= token_chars_mask.long()
    return np.stack([token_ids, chars], axis=2)


def get_raw_data(annotated_data, char_vocab, tag_vocab, feats_vocab):
    token_char_data, token_form_char_data, token_lemma_char_data, token_morph_data = annotated_data
    # token_data:
    # [1 x max_xtoken_len,
    #  [sent_id, xtoken, length]]
    # token_char_data:
    # [1 x max_token_char_len,
    #  [sent_id, token_id, morph_id, char, length]]
    # token_form_char_data:
    # [1 x max_form_char_len x
    #  [sent_id, token_id, morph_id, char, length, form_length, max_num_tokens, max_form_len]]
    # token_lemma_char_data:
    # [1 x max_lemma_char_len x
    #  [sent_id, token_id, morph_id, char, length, lemma_length, max_num_tokens, max_lemma_len]]
    # token_morph_data:
    # [1 x max_morph_len x
    #  [sent_id, token_id, morph_id, tag, feats, length, morph_length, max_num_tokens, max_morph_len]]
    seg_rows = _to_segmented_rows(token_char_data, token_form_char_data, token_lemma_char_data,
                                  token_morph_data, char_vocab, tag_vocab, feats_vocab)
    # token_id_rows = [r[0] for r in seg_rows]
    # token_rows = [r[1] for r in seg_rows]
    # form_rows = [r[2] for r in seg_rows]
    # start_offset_rows = list(range(len(seg_rows)))
    # end_offset_rows = list(range(1, len(seg_rows) + 1))
    # sent_id_rows = [sent_id.item()] * len(seg_rows)
    # is_gold_rows = [True] * len(seg_rows)
    #
    #     for sid, start, end, form, tid, token, is_gold in zip(sent_id_rows, start_offset_rows, end_offset_rows,
    #                                                           form_rows, token_id_rows, token_rows, is_gold_rows):
    #         lattice_rows.append([sid, start, end, form, '_', '_', '_', tid, token, is_gold])
    columns = ['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token', 'is_gold']
    # return pd.DataFrame(seg_rows, columns=columns)
    return  seg_rows


if __name__ == '__main__':
    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    # roberta_tokenizer = AutoTokenizer.from_pretrained("./experiments/transformers/roberta-bpe-byte-v1")
    # logging.info(f'{type(roberta_tokenizer).__name__} loaded')
    bert_tokenizer = AutoTokenizer.from_pretrained("./experiments/transformers/bert-wordpiece-v1")
    logging.info(f'{type(bert_tokenizer).__name__} loaded')
    # partition = tb.spmrl('data/raw/HebrewTreebank')
    partition = tb.spmrl_conllu_ner('data/raw/for_amit_spmrl')
    # partition = {k: v for k, v in partition.items() if k in ['dev', 'test']}
    # _save_emb(partition)
    char_vectors, char2index = _load_char_emb()
    # _save_samples(partition, bert_tokenizer, char2index, Path('data/processed/HebrewTreebank/hebtb'))
    _save_samples(partition, bert_tokenizer, char2index, Path('data/processed/for_amit_spmrl/hebtb'))
    # _save_data(partition, bert_tokenizer, char2index, Path('data/processed/HebrewTreebank/hebtb'))
    _save_data(partition, bert_tokenizer, char2index, Path('data/processed/for_amit_spmrl/hebtb'))
    # load_data(partition, bert_tokenizer)
