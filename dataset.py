import pandas as pd
import numpy as np
from pathlib import Path
import logging
from transformers import AutoTokenizer
from bclm import treebank as tb
import fasttext_emb as ft


def _xtokenize_tokens(xtokenizer, raw_df):
    df = raw_df[['sent_id', 'token_id', 'token']].drop_duplicates()
    return pd.DataFrame([(row.sent_id, row.token_id, row.token, xt)
                         for index, row in df.iterrows()
                         for xt in xtokenizer.tokenize(row.token)],
                        columns=['sent_id', 'token_id', 'token', 'xtoken'])


def _xtokenize_forms(xtokenizer, raw_df):
    rows = []
    raw_gb = raw_df.groupby(raw_df.sent_id)
    for sent_id, df in raw_gb:
        for form_id, row in enumerate(df.itertuples()):
            for xf in xtokenizer.tokenize(row.form):
                rows.append([row.sent_id, row.token_id, row.token, form_id + 1, row.form, xf])
    return pd.DataFrame(rows,
                        columns=['sent_id', 'token_id', 'token', 'form_id', 'form', 'xform'])


def _xtoken_to_char_data(xtoken_df):
    char_data_rows = []
    xtoken_gb = xtoken_df.groupby(xtoken_df.sent_id)
    for sent_id, df in xtoken_gb:
        for xtoken_id, row in enumerate(df.itertuples()):
            xtoken_chars = row.xtoken[2:] if row.xtoken[:2] == '##' else row.xtoken
            for c in xtoken_chars:
                char_data_rows.append([row.sent_id, row.token_id, row.token, xtoken_id + 1, row.xtoken, c])
    return pd.DataFrame(char_data_rows,
                        columns=["sent_id", "token_id", "token", "xtoken_id", "xtoken", "char"])


def _xform_to_char_data(xform_df):
    char_data_rows = []
    xform_gb = xform_df.groupby(xform_df.sent_id)
    for sent_id, df in xform_gb:
        for xform_id, row in enumerate(df.itertuples()):
            xform_chars = row.xform[2:] if row.xform[:2] == '##' else row.xform
            for c in xform_chars:
                char_data_rows.append([row.sent_id, row.token_id, row.token, row.form_id, row.form, xform_id + 1, row.xform, c])
    return pd.DataFrame(char_data_rows,
                        columns=["sent_id", "token_id", "token", "form_id", "form", "xform_id", "xform", "char"])


def _convert_xtokens_to_ids(df, xtokenizer):
    xtoken_ids = xtokenizer.convert_tokens_to_ids(df.xtoken)
    return [xtokenizer.cls_token_id] + xtoken_ids + [xtokenizer.sep_token_id]


def _convert_chars_to_ids(df, char_vocab):
    return [char_vocab[c] for c in df.char.tolist()]


class DataSample:

    def __init__(self, raw_df,
                 processed_token_df, processed_form_df,
                 processed_token_char_df, processed_form_char_df):
        raw_sent_id = raw_df.sent_id.unique()
        token_sent_id = processed_token_df.sent_id.unique()
        form_sent_id = processed_form_df.sent_id.unique()
        token_char_sent_id = processed_token_char_df.sent_id.unique()
        form_char_sent_id = processed_form_char_df.sent_id.unique()
        assert raw_sent_id[0] == token_sent_id[0]
        assert raw_sent_id[0] == form_sent_id[0]
        assert raw_sent_id[0] == token_char_sent_id[0]
        assert raw_sent_id[0] == form_char_sent_id[0]
        self.raw_df = raw_df
        self.processed_token_df = processed_token_df
        self.processed_form_df = processed_form_df
        self.processed_token_char_df = processed_token_char_df
        self.processed_form_char_df = processed_form_char_df


class ProcessedDataSamples:

    def __init__(self, xtokenizer, char_vocab, raw_df,
                 processed_token_df, processed_form_df,
                 processed_token_char_df, processed_form_char_df):
        self.xtokenizer = xtokenizer
        self.char_vocab = char_vocab
        self.raw_df = raw_df
        self.processed_token_df = processed_token_df
        self.processed_form_df = processed_form_df
        self.processed_token_char_df = processed_token_char_df
        self.processed_form_char_df = processed_form_char_df

    def save(self, partition_name, root_path):
        token_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_token.csv'
        logging.info(f'saving {token_file}')
        self.processed_token_df.to_csv(str(token_file))
        form_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_form.csv'
        logging.info(f'saving {form_file}')
        self.processed_form_df.to_csv(str(form_file))
        token_char_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_token_char.csv'
        logging.info(f'saving {token_char_file}')
        self.processed_token_char_df.to_csv(str(token_char_file))
        form_char_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_form_char.csv'
        logging.info(f'saving {form_char_file}')
        self.processed_form_char_df.to_csv(str(form_char_file))

    def to_data(self):
        samples = list(self)
        # samples = self[:10]
        xtoken_data = self._to_xtoken_data(samples)
        token_char_data = self._to_token_char_data(samples)
        form_char_data = self._to_form_char_data(samples)
        token_form_char_data = self._to_token_form_char_data(form_char_data)
        return xtoken_data, token_char_data, form_char_data, token_form_char_data

    def _to_xtoken_data(self, samples):
        max_len = max([len(sample.processed_token_df.xtoken) + 2 for sample in samples])
        sent_ids, xtokens = [], []
        for sample in samples:
            sample_xtokens = [self.xtokenizer.cls_token] + sample.processed_token_df.xtoken.tolist() + [self.xtokenizer.sep_token]
            sample_sent_id = [sample.processed_token_df.sent_id.unique()[0] for _ in sample_xtokens]
            sample_xtokens += [self.xtokenizer.pad_token] * (max_len - len(sample_xtokens))
            sample_sent_id += [sample_sent_id[0]] * (max_len - len(sample_sent_id))
            sent_ids.extend(sample_sent_id)
            xtokens.extend(sample_xtokens)
        return pd.DataFrame(zip(sent_ids, xtokens),
                            columns=['sent_id', 'xtoken'])

    def _to_token_char_data(self, samples):
        max_len = max([len(sample.processed_token_char_df.char) for sample in samples])
        sent_ids, tokens, token_ids, xtokens, xtoken_ids, chars = [], [], [], [], [], []
        for sample in samples:
            sample_chars = sample.processed_token_char_df.char.tolist()
            sample_tokens = sample.processed_token_char_df.token.tolist()
            sample_token_ids = sample.processed_token_char_df.token_id.tolist()
            sample_xtokens = sample.processed_token_char_df.xtoken.tolist()
            sample_xtoken_ids = sample.processed_token_char_df.xtoken_id.tolist()
            sample_sent_id = [sample.processed_token_char_df.sent_id.unique()[0] for _ in sample_chars]
            sample_chars += ['<pad>'] * (max_len - len(sample_chars))
            sample_tokens += ['<pad>'] * (max_len - len(sample_tokens))
            sample_token_ids += [-1] * (max_len - len(sample_token_ids))
            sample_xtokens += [self.xtokenizer.pad_token] * (max_len - len(sample_xtokens))
            sample_xtoken_ids += [-1] * (max_len - len(sample_xtoken_ids))
            sample_sent_id += [sample_sent_id[0]] * (max_len - len(sample_sent_id))
            sent_ids.extend(sample_sent_id)
            tokens.extend(sample_tokens)
            token_ids.extend(sample_token_ids)
            xtokens.extend(sample_xtokens)
            xtoken_ids.extend(sample_xtoken_ids)
            chars.extend(sample_chars)
        return pd.DataFrame(zip(sent_ids, tokens, token_ids, xtokens, xtoken_ids, chars),
                            columns=['sent_id', 'token', 'token_id', 'xtoken', 'xtoken_id', 'char'])

    def _to_form_char_data(self, samples):
        processed_samples = []
        for sample in samples:
            sample_chars = sample.processed_form_char_df.char.tolist()
            sample_tokens = sample.processed_form_char_df.token.tolist()
            sample_token_ids = sample.processed_form_char_df.token_id.tolist()
            sample_forms = sample.processed_form_char_df.form.tolist()
            sample_form_ids = sample.processed_form_char_df.form_id.to_numpy()
            form_pos = sample_form_ids[:-1] != sample_form_ids[1:]
            form_pos = np.where(form_pos)[0]
            sample_form_ids = sample_form_ids.tolist()
            for i, pos in enumerate(form_pos):
                sample_chars.insert(pos + i + 1, '<sep>')
                sample_tokens.insert(pos + i + 1, sample_tokens[pos + i])
                sample_token_ids.insert(pos + i + 1, sample_token_ids[pos + i])
                sample_forms.insert(pos + i + 1, sample_forms[pos + i])
                sample_form_ids.insert(pos + i + 1, sample_form_ids[pos + i])
            sample_token_ids = np.array(sample_token_ids)
            token_pos = sample_token_ids[:-1] != sample_token_ids[1:]
            token_pos = np.where(token_pos)[0]
            sample_token_ids = sample_token_ids.tolist()
            for pos in token_pos:
                sample_chars[pos] = '</s>'
            # sample_chars.insert(0, '<s>')
            # sample_tokens.insert(0, '<s>')
            # sample_token_ids.insert(0, 0)
            # sample_forms.insert(0, '<s>')
            # sample_form_ids.insert(0, 0)
            sample_chars.append('</s>')
            sample_tokens.append('</s>')
            sample_token_ids.append(sample_token_ids[-1])
            sample_forms.append('</s>')
            sample_form_ids.append(sample_form_ids[-1])
            sample_sent_id = [sample.processed_form_char_df.sent_id.unique()[0] for _ in sample_chars]
            processed_samples.append((sample_sent_id, sample_tokens, sample_token_ids, sample_forms, sample_form_ids, sample_chars))

        sent_ids, tokens, token_ids, forms, form_ids, chars = [], [], [], [], [], []
        max_len = max([len(sample[0]) for sample in processed_samples])
        for sample in processed_samples:
            sample_sent_id, sample_tokens, sample_token_ids, sample_forms, sample_form_ids, sample_chars = sample
            sample_chars += ['<pad>'] * (max_len - len(sample_chars))
            sample_tokens += ['<pad>'] * (max_len - len(sample_tokens))
            sample_token_ids += [-1] * (max_len - len(sample_token_ids))
            sample_forms += ['<pad>'] * (max_len - len(sample_forms))
            sample_form_ids += [-1] * (max_len - len(sample_form_ids))
            sample_sent_id += [sample_sent_id[0]] * (max_len - len(sample_sent_id))
            sent_ids.extend(sample_sent_id)
            tokens.extend(sample_tokens)
            token_ids.extend(sample_token_ids)
            forms.extend(sample_forms)
            form_ids.extend(sample_form_ids)
            chars.extend(sample_chars)
        return pd.DataFrame(zip(sent_ids, tokens, token_ids, forms, form_ids, chars),
                            columns=['sent_id', 'token', 'token_id', 'form', 'form_id', 'char'])

    def _to_token_form_char_data(self, form_char_data):
        rows = []
        gb = form_char_data.groupby(['sent_id', 'token_id'])
        max_form_len = max([len(group) for key, group in gb if key[1] != -1])
        max_num_tokens = max([key[1] for key, group in gb])
        sent_gb = form_char_data.groupby(['sent_id'])
        for sent_id, sent_group in sent_gb:
            token_gb = sent_group.groupby('token_id')
            sent_rows = []
            token_id = 0
            for token_id, token_group in token_gb:
                if token_id == -1:
                    continue
                group_rows = token_group.to_numpy().tolist()
                for row in group_rows:
                    row.extend([max_num_tokens, max_form_len])
                if token_id == 0:
                    sent_rows.extend(group_rows)
                else:
                    fill_row = [sent_id, token_group.token.unique()[0], token_id, '<pad>', -1, '<pad>', max_num_tokens, max_form_len]
                    fill_len = max_form_len - len(group_rows)
                    fill_rows = [fill_row] * fill_len
                    sent_rows.extend(group_rows)
                    sent_rows.extend(fill_rows)
            fill_row = [sent_id, '<pad>', -1, '<pad>', -1, '<pad>', max_num_tokens, max_form_len]
            fill_len = max_form_len * (max_num_tokens - token_id)
            fill_rows = [fill_row] * fill_len
            sent_rows.extend(fill_rows)
            rows.extend(sent_rows)
        return pd.DataFrame(rows, columns=['sent_id', 'token', 'token_id', 'form', 'form_id', 'char', 'max_num_tokens', 'max_form_len'])

    def __len__(self):
        raw_gb = self.raw_df.groupby(self.raw_df.sent_id)
        return len(raw_gb)

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            return [self[i + 1] for i in indices]
        sent_raw_df = self.raw_df[self.raw_df.sent_id == key]
        sent_processed_token_df = self.processed_token_df[self.processed_token_df.sent_id == key]
        sent_processed_form_df = self.processed_form_df[self.processed_form_df.sent_id == key]
        sent_processed_token_char_df = self.processed_token_char_df[self.processed_token_char_df.sent_id == key]
        sent_processed_form_char_df = self.processed_form_char_df[self.processed_form_char_df.sent_id == key]
        return DataSample(sent_raw_df,
                          sent_processed_token_df, sent_processed_form_df,
                          sent_processed_token_char_df, sent_processed_form_char_df)

    def __iter__(self):
        raw_gb = self.raw_df.groupby(self.raw_df.sent_id)
        for sent_id, sent_raw_df in raw_gb:
            sent_processed_token_df = self.processed_token_df[self.processed_token_df.sent_id == sent_id]
            sent_processed_form_df = self.processed_form_df[self.processed_form_df.sent_id == sent_id]
            sent_processed_token_char_df = self.processed_token_char_df[self.processed_token_char_df.sent_id == sent_id]
            sent_processed_form_char_df = self.processed_form_char_df[self.processed_form_char_df.sent_id == sent_id]
            yield DataSample(sent_raw_df,
                             sent_processed_token_df, sent_processed_form_df,
                             sent_processed_token_char_df, sent_processed_form_char_df)

    class Builder:

        def __init__(self, xtokenizer, char_vocab):
            self.xtokenizer = xtokenizer
            self.char_vocab = char_vocab

        def get_samples(self, raw_df, partition_name):
            logging.info(f'processing {partition_name} {type(self.xtokenizer).__name__} tokens')
            processed_token_df = _xtokenize_tokens(self.xtokenizer, raw_df)
            logging.info(f'processing {partition_name} {type(self.xtokenizer).__name__} forms')
            processed_form_df = _xtokenize_forms(self.xtokenizer, raw_df)
            logging.info(f'processing {partition_name} {type(self.xtokenizer).__name__} token chars')
            processed_token_char_df = _xtoken_to_char_data(processed_token_df)
            logging.info(f'processing {partition_name} {type(self.xtokenizer).__name__} form chars')
            processed_form_char_df = _xform_to_char_data(processed_form_df)
            return ProcessedDataSamples(self.xtokenizer, self.char_vocab, raw_df,
                                        processed_token_df, processed_form_df,
                                        processed_token_char_df, processed_form_char_df)

        def load_samples(self, raw_df, partition_name, root_path):
            token_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_token.csv'
            logging.info(f'loading {token_file}')
            processed_token_df = pd.read_csv(token_file, index_col=0)
            form_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_form.csv'
            logging.info(f'loading {form_file}')
            processed_form_df = pd.read_csv(form_file, index_col=0)
            token_char_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_token_char.csv'
            logging.info(f'loading {token_char_file}')
            processed_token_char_df = pd.read_csv(token_char_file, index_col=0)
            form_char_file = root_path / f'{partition_name}_{type(self.xtokenizer).__name__}_form_char.csv'
            logging.info(f'loading {form_char_file}')
            processed_form_char_df = pd.read_csv(form_char_file, index_col=0)
            return ProcessedDataSamples(self.xtokenizer, self.char_vocab, raw_df,
                                        processed_token_df, processed_form_df,
                                        processed_token_char_df, processed_form_char_df)


def _save_emb(partition):
    tokens = set(token for part in partition for token in partition[part].token)
    forms = set(form for part in partition for form in partition[part].form)
    chars = set(c.lower() for word in list(forms) + list(tokens) for c in word)
    tokens = ['<pad>', '<sep>', '<s>', '</s>'] + sorted(list(tokens))
    forms = ['<pad>', '<sep>', '<s>', '</s>'] + sorted(list(forms))
    chars = ['<pad>', '<sep>', '<s>', '</s>'] + sorted(list(chars))
    token_vectors, token2index = ft.get_word_vectors('he', Path('/Users/Amit/dev/fastText/models/cc.he.300.bin'), tokens)
    form_vectors, form2index = ft.get_word_vectors('he', Path('/Users/Amit/dev/fastText/models/cc.he.300.bin'), forms)
    char_vectors, char2index = ft.get_word_vectors('he', Path('/Users/Amit/dev/fastText/models/cc.he.300.bin'), chars)
    ft.save_word_vectors(Path('data/ft_token.vec.txt'), token_vectors, token2index)
    ft.save_word_vectors(Path('data/ft_form.vec.txt'), form_vectors, form2index)
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
        xtoken_data, token_char_data, form_char_data, token_form_char_data = samples.to_data()
        token_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_xtoken.csv'
        logging.info(f'saving {token_file}')
        xtoken_data.to_csv(str(token_file))
        token_char_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_token_char.csv'
        logging.info(f'saving {token_char_file}')
        token_char_data.to_csv(str(token_char_file))
        form_char_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_form_char.csv'
        logging.info(f'saving {form_char_file}')
        form_char_data.to_csv(str(form_char_file))
        token_form_char_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_token_form_char.csv'
        logging.info(f'saving {token_form_char_file}')
        token_form_char_data.to_csv(str(token_form_char_file))


def load_emb(data_type):
    return ft.load_word_vectors(Path(f'data/ft_{data_type}.vec.txt'))


def load_data(partition, xtokenizer, char_vocab):
    data = {}
    for part in partition:
        token_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_xtoken.csv'
        logging.info(f'loading {token_file}')
        xtoken_data = pd.read_csv(str(token_file), index_col=0)
        xtoken_data['xtoken'] = xtoken_data['xtoken'].apply(lambda x: xtokenizer.convert_tokens_to_ids(x))
        xtoken_groups = xtoken_data.groupby('sent_id')
        # logging.info(f'{part} xtoken samples # = {len(xtoken_groups)}')
        xtoken_data = np.stack([sent_df.to_numpy() for _, sent_df in xtoken_groups], axis=0)

        token_char_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_token_char.csv'
        logging.info(f'loading {token_char_file}')
        token_char_data = pd.read_csv(str(token_char_file), index_col=0)
        token_char_data['char'] = token_char_data['char'].apply(lambda x: char_vocab['char2index'][x])
        token_char_data['xtoken'] = token_char_data['xtoken'].apply(lambda x: xtokenizer.convert_tokens_to_ids(x))
        token_char_data = token_char_data[['sent_id', 'token_id', 'xtoken_id', 'char']]
        token_char_groups = token_char_data.groupby('sent_id')
        # logging.info(f'{part} token char samples # = {len(token_char_groups)}')
        token_char_data = np.stack([sent_df.to_numpy() for sent_id, sent_df in token_char_groups], axis=0)

        form_char_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_form_char.csv'
        logging.info(f'loading {form_char_file}')
        form_char_data = pd.read_csv(str(form_char_file), index_col=0)
        form_char_data['char'] = form_char_data['char'].apply(lambda x: char_vocab['char2index'][x])
        form_char_data = form_char_data[['sent_id', 'char']]
        form_char_groups = form_char_data.groupby('sent_id')
        # logging.info(f'{part} form char samples # = {len(form_char_groups)}')
        form_char_data = np.stack([sent_df.to_numpy() for sent_id, sent_df in form_char_groups], axis=0)

        token_form_char_file = Path('data') / f'{part}_{type(xtokenizer).__name__}_token_form_char.csv'
        logging.info(f'loading {token_form_char_file}')
        token_form_char_data = pd.read_csv(str(token_form_char_file), index_col=0)
        token_form_char_data['char'] = token_form_char_data['char'].apply(lambda x: char_vocab['char2index'][x])
        token_form_char_data = token_form_char_data[['sent_id', 'char', 'max_num_tokens', 'max_form_len']]
        token_form_char_groups = token_form_char_data.groupby(['sent_id'])
        # logging.info(f'{part} token form char samples # = {len(token_form_char_groups)}')
        token_form_char_data = np.stack([sent_df.to_numpy() for _, sent_df in token_form_char_groups], axis=0)

        data[part] = (xtoken_data, token_char_data, form_char_data, token_form_char_data)
    return data


def _to_segmented_rows(input_token_chars, sent_form_chars, char_vocab):
    sep_char = char_vocab['char2index']['<sep>']
    token_ids = sent_form_chars[:, :, 0]
    # token_ids = token_ids[token_ids != 0]
    num_tokens = token_ids[:, -1].item()
    rows = []
    for token_id in range(1, num_tokens+1):
        tokens = input_token_chars[input_token_chars[:, :, 0] == token_id]
        token_chars = tokens[:, 1]
        chars = [char_vocab['index2char'][c] for c in token_chars]
        token = ''.join(chars)
        segments = sent_form_chars[sent_form_chars[:, :, 0] == token_id]
        segments_chars = segments[:, 1]
        chars = [' ' if c == sep_char else char_vocab['index2char'][c] for c in segments_chars
                 if c != char_vocab['char2index']['</s>']]
        rows.extend([(token_id, token, seg) for seg in ''.join(chars).split(' ')])
    return rows


def to_raw_data(seg_data, char_vocab):
    lattice_rows = []
    for data in seg_data:
        sent_ids, token_chars, form_chars = data
        sent_id = sent_ids[0, 0]
        seg_rows = _to_segmented_rows(token_chars, form_chars, char_vocab)

        token_id_rows = [r[0] for r in seg_rows]
        token_rows = [r[1] for r in seg_rows]
        form_rows = [r[2] for r in seg_rows]
        start_offset_rows = list(range(len(seg_rows)))
        end_offset_rows = list(range(1, len(seg_rows) + 1))
        sent_id_rows = [sent_id.item()] * len(seg_rows)
        is_gold_rows = [True] * len(seg_rows)

        for sid, start, end, form, tid, token, is_gold in zip(sent_id_rows, start_offset_rows, end_offset_rows,
                                                              form_rows, token_id_rows, token_rows, is_gold_rows):
            lattice_rows.append([sid, start, end, form, '_', '_', '_', tid, token, is_gold])
    return pd.DataFrame(lattice_rows, columns=['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats',
                                               'token_id', 'token', 'is_gold'])


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
    partition = tb.spmrl('data/raw')
    # partition = ['train', 'dev', 'test']
    # _save_emb(partition)
    char_vectors, char2index = load_emb('char')
    # _save_samples(partition, bert_tokenizer, char2index, Path('data/processed/hebtb'))
    _save_data(partition, bert_tokenizer, char2index, Path('data/processed/hebtb'))
    # load_data(partition, bert_tokenizer)
