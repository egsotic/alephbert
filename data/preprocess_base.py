from transformers import BertTokenizerFast
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import fasttext_emb as ft
from pathlib import Path


def add_chars(df: pd.DataFrame, field_name):
    field_id = df.columns.get_loc(field_name)
    rows = [[list(row[1:]) + [c] for c in row[1:][field_id]] for row in df.itertuples()]
    rows = [char_row for word_rows in rows for char_row in word_rows]
    return pd.DataFrame(rows, columns=list(df.columns) + ['char'])


def _create_xtoken_df(morph_df: pd.DataFrame, xtokenizer: BertTokenizerFast):
    token_df = morph_df[['sent_id', 'token_id', 'token']].drop_duplicates()
    sent_groups = list(token_df.groupby([token_df.sent_id]))
    num_sentences = len(sent_groups)
    tq = tqdm(total=num_sentences, desc="Sentence")
    data_rows = []
    for sent_id, sent_df in sent_groups:
        xtokens = [(tid, t, xt) for tid, t in zip(sent_df.token_id, sent_df.token) for xt in xtokenizer.tokenize(t)]
        token_ids = [0] + [tid for tid, t, xt in xtokens] + [sent_df.token_id.max() + 1]
        tokens = ['<s>'] + [t for tid, t, xt in xtokens] + ['</s>']
        xtokens = [xtokenizer.cls_token] + [xt for tid, t, xt in xtokens] + [xtokenizer.sep_token]
        sent_ids = [sent_id] * len(xtokens)
        data_rows.extend(list(zip(sent_ids, token_ids, tokens, xtokens)))
        tq.update(1)
    tq.close()
    return pd.DataFrame(data_rows, columns=['sent_id', 'token_id', 'token', 'xtoken'])


def _create_token_char_df(morph_df: pd.DataFrame):
    char_df = morph_df[['sent_id', 'token_id', 'token']].drop_duplicates()
    return add_chars(char_df, 'token')


def _collate_xtoken_data_samples(xtoken_df: pd.DataFrame, xtokenizer: BertTokenizerFast):
    sent_groups = list(xtoken_df.groupby(xtoken_df.sent_id))
    num_sentences = len(sent_groups)
    max_sent_len = max([len(sent_df) for sent_id, sent_df in sent_groups])
    data_rows = []
    tq = tqdm(total=num_sentences, desc="Sentence")
    for sent_id, sent_df in sent_groups:
        sent_idxs = list(sent_df.sent_id)
        token_idxs = list(sent_df.token_id)
        tokens = list(sent_df.token)
        xtokens = list(sent_df.xtoken)
        xtoken_ids = xtokenizer.convert_tokens_to_ids(xtokens)
        pad_len = max_sent_len - len(sent_idxs)
        sent_idxs.extend([sent_id] * pad_len)
        tokens.extend(['<pad>'] * pad_len)
        token_idxs.extend([-1] * pad_len)
        xtokens.extend([xtokenizer.pad_token] * pad_len)
        xtoken_ids.extend([xtokenizer.pad_token_id] * pad_len)
        data_rows.extend(list(row) for row in zip(sent_idxs, token_idxs, tokens, xtokens, xtoken_ids))
        tq.update(1)
    tq.close()
    return pd.DataFrame(data_rows, columns=['sent_idx', 'token_idx', 'token', 'xtoken', 'xtoken_id'])


def _collate_token_char_data_samples(token_char_df: pd.DataFrame, char2index: dict):
    sent_groups = list(token_char_df.groupby(token_char_df.sent_id))
    num_sentences = len(sent_groups)
    max_sent_len = max([len(sent_df) for sent_id, sent_df in sent_groups])
    data_rows = []
    tq = tqdm(total=num_sentences, desc="Sentence")
    for sent_id, sent_df in sent_groups:
        sent_idxs = list(sent_df.sent_id)
        token_idxs = list(sent_df.token_id)
        tokens = list(sent_df.token)
        chars = list(sent_df.char)
        char_ids = [char2index[c] for c in chars]
        pad_len = max_sent_len - len(sent_idxs)
        sent_idxs.extend([sent_id] * pad_len)
        tokens.extend(['<pad>'] * pad_len)
        token_idxs.extend([-1] * pad_len)
        chars.extend(['<pad>'] * pad_len)
        char_ids.extend([char2index['<pad>']] * pad_len)
        data_rows.extend(list(row) for row in zip(sent_idxs, token_idxs, tokens, chars, char_ids))
        tq.update(1)
    tq.close()
    return pd.DataFrame(data_rows, columns=['sent_idx', 'token_idx', 'token', 'char', 'char_id'])


def save_char_vocab(data_path: Path, ft_root_path: Path, raw_partition: dict):
    logging.info(f'saving char embedding')
    tokens = set(token for part in raw_partition for token in raw_partition[part].token)
    forms = set(token for part in raw_partition for token in raw_partition[part].form)
    lemmas = set(token for part in raw_partition for token in raw_partition[part].lemma)
    chars = set(c.lower() for word in list(tokens) + list(forms) + list(lemmas) for c in word)
    chars = ['<pad>', '<sep>', '<s>', '</s>'] + sorted(list(chars))
    char_vectors, char2index = ft.get_word_vectors('he', ft_root_path / 'models/cc.he.300.bin', chars)
    ft.save_word_vectors(data_path / 'ft_char.vec.txt', char_vectors, char2index)


def load_char_vocab(data_path: Path):
    logging.info(f'loading char embedding')
    char_vectors, char2index = ft.load_word_vectors(data_path / 'ft_char.vec.txt')
    index2char = {char2index[c]: c for c in char2index}
    char_vocab = {'char2index': char2index, 'index2char': index2char}
    return char_vectors, char_vocab


def get_token_char_data(data_path: Path, morph_partition: dict):
    token_char_partition = {}
    for part in morph_partition:
        token_char_file = data_path / f'{part}_token_char.csv'
        if not token_char_file.exists():
            logging.info(f'processing {part} token chars')
            token_char_df = _create_token_char_df(morph_partition[part])
            logging.info(f'saving {token_char_file}')
            token_char_df.to_csv(str(token_char_file))
        else:
            logging.info(f'loading {token_char_file}')
            token_char_df = pd.read_csv(str(token_char_file), index_col=0)
        token_char_partition[part] = token_char_df
    return token_char_partition


def get_xtoken_data(data_path: Path, morph_partition: dict, xtokenizer: BertTokenizerFast):
    xtoken_partition = {}
    for part in morph_partition:
        xtoken_file = data_path / f'{part}_xtoken.csv'
        if not xtoken_file.exists():
            logging.info(f'processing {part} xtokens')
            xtoken_df = _create_xtoken_df(morph_partition[part], xtokenizer)
            logging.info(f'saving {xtoken_file}')
            xtoken_df.to_csv(str(xtoken_file))
        else:
            logging.info(f'loading {xtoken_file}')
            xtoken_df = pd.read_csv(str(xtoken_file), index_col=0)
        xtoken_partition[part] = xtoken_df
    return xtoken_partition


def save_xtoken_data_samples(data_path: Path, xtoken_partition: dict, xtokenizer: BertTokenizerFast):
    xtoken_samples_partition = {}
    for part in xtoken_partition:
        xtoken_samples_file = data_path / f'{part}_xtoken_data_samples.csv'
        logging.info(f'processing {part} xtoken data samples')
        samples_df = _collate_xtoken_data_samples(xtoken_partition[part], xtokenizer)
        logging.info(f'saving {xtoken_samples_file}')
        samples_df.to_csv(str(xtoken_samples_file))
        xtoken_samples_partition[part] = samples_df
    return xtoken_samples_partition


def load_xtoken_data_samples(data_path: Path, partition: list):
    xtoken_samples_partition = {}
    for part in partition:
        xtoken_samples_file = data_path / f'{part}_xtoken_data_samples.csv'
        logging.info(f'loading {xtoken_samples_file}')
        samples_df = pd.read_csv(str(xtoken_samples_file), index_col=0)
        xtoken_samples_partition[part] = samples_df
    return xtoken_samples_partition


def save_token_char_data_samples(data_path: Path, token_partition: dict, char2index: dict):
    token_char_samples_partition = {}
    for part in token_partition:
        token_char_samples_file = data_path / f'{part}_token_char_data_samples.csv'
        logging.info(f'processing {part} token char data samples')
        samples_df = _collate_token_char_data_samples(token_partition[part], char2index)
        logging.info(f'saving {token_char_samples_file}')
        samples_df.to_csv(str(token_char_samples_file))
        token_char_samples_partition[part] = samples_df
    return token_char_samples_partition


def load_token_char_data_samples(data_path: Path, partition: list):
    token_char_samples_partition = {}
    for part in partition:
        token_char_samples_file = data_path / f'{part}_token_char_data_samples.csv'
        logging.info(f'loading {token_char_samples_file}')
        samples_df = pd.read_csv(str(token_char_samples_file), index_col=0)
        token_char_samples_partition[part] = samples_df
    return token_char_samples_partition


def load_token_data(data_path: Path, partition: list):
    xtoken_data_samples = load_xtoken_data_samples(data_path, partition)
    token_char_data_samples = load_token_char_data_samples(data_path, partition)
    arr_data = {}
    for part in partition:
        xtoken_df = xtoken_data_samples[part]
        token_data = xtoken_df[['sent_idx', 'token_idx', 'xtoken_id']]
        token_data_groups = token_data.groupby('sent_idx')
        token_arr = np.stack([sent_df.to_numpy() for sent_id, sent_df in token_data_groups], axis=0)

        token_char_df = token_char_data_samples[part]
        token_char_data = token_char_df[['sent_idx', 'token_idx', 'char_id']]
        token_char_data_groups = token_char_data.groupby('sent_idx')
        token_char_arr = np.stack([sent_df.to_numpy() for sent_id, sent_df in token_char_data_groups], axis=0)

        arr_data[part] = (token_arr, token_char_arr)
    return arr_data
