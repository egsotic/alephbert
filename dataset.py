import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from transformers import AutoTokenizer
from bclm import treebank as tb
import fasttext_emb as ft
from torch.utils.data import TensorDataset


def _xtokenize_tokens(partition, xtokenizer):
    xdata = {}
    for part in partition:
        df = partition[part]
        logging.info(f'xtokenizing {part} {type(xtokenizer).__name__} tokens')
        raw_df = df[['sent_id', 'token_id', 'token']].drop_duplicates()
        xdf = pd.DataFrame([(row.sent_id, row.token_id, row.token, xt)
                            for index, row in raw_df.iterrows()
                            for xt in xtokenizer.tokenize(row.token)],
                           columns=['sent_id', 'token_id', 'token', 'xtoken'])
        xdata[part] = xdf
    return xdata


def _save_xdata(data_root_path, xdata, xtokenizer, data_type):
    for part in xdata:
        xdf = xdata[part]
        xdata_file = Path(data_root_path) / f'{part}_{type(xtokenizer).__name__}_{data_type}.csv'
        logging.info(f'saving: {xdata_file}')
        xdf.to_csv(xdata_file)


def _load_xdata(data_root_path, partition, xtokenizer, data_type):
    xdata = {}
    for part in partition:
        xdata_file = Path(data_root_path) / f'{part}_{type(xtokenizer).__name__}_{data_type}.csv'
        logging.info(f'loading: {xdata_file}')
        xdf = pd.read_csv(xdata_file, index_col=0)
        xdata[part] = xdf
    return xdata


def _get_forms(partition):
    data = {}
    for part in partition:
        df = partition[part]
        raw_df = df[['sent_id', 'token_id', 'form']]
        data[part] = raw_df
    return data


def _save_data(data_root_path, data, data_type):
    for part in data:
        df = data[part]
        data_file = Path(data_root_path) / f'{part}_{data_type}.csv'
        logging.info(f'saving: {data_file}')
        df.to_csv(data_file)


def _load_data(data_root_path, partition, data_type):
    data = {}
    for part in partition:
        data_file = Path(data_root_path) / f'{part}_{data_type}.csv'
        logging.info(f'loading: {data_file}')
        df = pd.read_csv(data_file, index_col=0)
        data[part] = df
    return data


def _form_to_char_data(form_data):
    char_data = {}
    for part in form_data:
        part_df = form_data[part]
        char_data_rows = []
        part_gb = part_df.groupby(part_df.sent_id)
        for df in [part_gb.get_group(x).reset_index(drop=True) for x in part_gb.groups]:
            for form_id, row in enumerate(df.itertuples()):
                for c in row.form:
                    char_data_rows.append([row.sent_id, row.token_id, form_id + 1, row.form, c])
        char_df = pd.DataFrame(char_data_rows, columns=["sent_id", "token_id", "form_id", "form", "char"])
        char_data[part] = char_df
    return char_data


def _convert_xtokens_to_ids(xtoken_df, xtokenizer, max_len):
    xtoken_ids = xtokenizer.convert_tokens_to_ids(xtoken_df.xtoken)
    xtoken_ids = [xtokenizer.cls_token_id] + xtoken_ids + [xtokenizer.sep_token_id]
    mask = [1] * len(xtoken_ids)
    fill_mask_len = max_len + 2 - len(xtoken_ids)
    xtoken_ids += [xtokenizer.pad_token_id] * fill_mask_len
    mask += [0] * fill_mask_len
    return xtoken_ids, mask


def _convert_form_chars_to_ids(form_char_df, form_char_vocab, max_len):
    char_ids = []
    form_gb = form_char_df.groupby(form_char_df.form_id)
    for form_id, g in form_gb:
        char_ids.extend([form_char_vocab[c] for c in g.char] + [form_char_vocab['<sep>']])
    char_ids = [form_char_vocab['<s>']] + char_ids + [form_char_vocab['</s>']]
    mask = [1] * len(char_ids)
    mask[0] = 0
    mask[-1] = 0
    fill_mask_len = max_len + 2 - len(char_ids)
    char_ids += [form_char_vocab['<pad>']] * fill_mask_len
    mask += [0] * fill_mask_len
    return char_ids, mask


def _to_form_data(xtoken_data, xtokenizer, form_char_data, form_char_vocab):
    data_samples = {}
    for part in xtoken_data:
        samples = []
        xtoken_df = xtoken_data[part]
        xtoken_gb = xtoken_df.groupby(xtoken_df.sent_id)
        max_xtoken_len = max([len(df) for sent_id, df in xtoken_gb])

        form_char_df = form_char_data[part]
        form_char_gb = form_char_df.groupby(form_char_df.sent_id)
        max_form_char_len = max([len(df) + df.form_id.unique()[-1] for sent_id, df in form_char_gb])
        for sent_id, out_df in form_char_gb:
            in_df = xtoken_gb.get_group(sent_id)
            in_sample = _convert_xtokens_to_ids(in_df, xtokenizer, max_xtoken_len)
            out_sample = _convert_form_chars_to_ids(out_df, form_char_vocab, max_form_char_len)
            samples.append((*in_sample, *out_sample))
        input_ids = np.array([sample[0] for sample in samples], dtype=np.int)
        input_mask = np.array([sample[1] for sample in samples], dtype=np.int)
        output_ids = np.array([sample[2] for sample in samples], dtype=np.int)
        output_mask = np.array([sample[3] for sample in samples], dtype=np.int)
        data_samples[part] = (input_ids, input_mask, output_ids, output_mask)
    return data_samples


def _to_tensor_dataset(data_samples):
    tensor_data = {}
    for part in data_samples:
        logging.info(f'transforming {part}')
        inputs_tensor = torch.tensor(data_samples[part][0], dtype=torch.long)
        inputs_mask_tensor = torch.tensor(data_samples[part][1], dtype=torch.long)
        outputs_tensor = torch.tensor(data_samples[part][2], dtype=torch.long)
        outputs_mask_tensor = torch.tensor(data_samples[part][3], dtype=torch.long)
        tensor_data[part] = TensorDataset(inputs_tensor, inputs_mask_tensor, outputs_tensor, outputs_mask_tensor)
    return tensor_data


def _save_tensor_dataset(data_root_path, tensor_data, xtokenizer):
    for part in tensor_data:
        data_file_path = Path(data_root_path) / f'{part}_{type(xtokenizer).__name__}_dataset.pt'
        logging.info(f'saving: {data_file_path}')
        torch.save(tensor_data[part], data_file_path)


def load_tensor_dataset(data_root_path, partition, xtokenizer):
    tensor_data = {}
    for part in partition:
        data_file_path = Path(data_root_path) / f'{part}_{type(xtokenizer).__name__}_dataset.pt'
        logging.info(f'loading: {data_file_path}')
        tensor_samples = torch.load(data_file_path)
        tensor_data[part] = tensor_samples
    return tensor_data


def _save_form_char_emb(partition):
    # tokens = set(token for token in partition['train'].token)
    # forms = set(form for form in partition['train'].form)
    forms = set(form for part in partition for form in partition[part].form)
    chars = set(c for form in forms for c in form)
    # tokens = ['<pad>', '<sep>', '<s>', '</s>'] + list(tokens)
    # forms = ['<pad>', '<sep>', '<s>', '</s>'] + list(forms)
    chars = ['<pad>', '<sep>', '<s>', '</s>'] + list(chars)
    # token_vectors, token2index = ft.get_word_vectors('he', Path('/Users/Amit/dev/fastText/models/cc.he.300.bin'), tokens)
    # form_vectors, form2index = ft.get_word_vectors('he', Path('/Users/Amit/dev/fastText/models/cc.he.300.bin'), forms)
    char_vectors, char2index = ft.get_word_vectors('he', Path('/Users/Amit/dev/fastText/models/cc.he.300.bin'), chars)
    # ft.save_word_vectors(Path('data/ft_token.vec.txt'), token_vectors, token2index)
    # ft.save_word_vectors(Path('data/ft_form.vec.txt'), form_vectors, form2index)
    ft.save_word_vectors(Path('data/ft_char.vec.txt'), char_vectors, char2index)


def load_form_char_emb():
    # token_vectors, token2index = ft.load_word_vectors(Path(f'data/ft_token.vec.txt'))
    # form_vectors, form2index = ft.load_word_vectors(Path(f'data/ft_form.vec.txt'))
    return ft.load_word_vectors(Path(f'data/ft_char.vec.txt'))


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
    # partition = {'train': None, 'dev': None, 'test': None}
    _save_form_char_emb(partition)
    char_vectors, char2index = load_form_char_emb()
    xtoken_data = _xtokenize_tokens(partition, bert_tokenizer)
    _save_xdata(Path('data/processed/hebtb'), xtoken_data, bert_tokenizer, 'token')
    form_data = _get_forms(partition)
    _save_data(Path('data/processed/hebtb'), form_data, 'form')
    form_data = _load_data(Path('data/processed/hebtb'), partition, 'form')
    char_data = _form_to_char_data(form_data)
    _save_data(Path('data/processed/hebtb'), char_data, 'char')
    xtoken_data = _load_xdata(Path('data/processed/hebtb'), partition, bert_tokenizer, 'token')
    char_data = _load_data(Path('data/processed/hebtb'), partition, 'char')
    data_samples = _to_form_data(xtoken_data, bert_tokenizer, char_data, char2index)
    tensor_dataset = _to_tensor_dataset(data_samples)
    _save_tensor_dataset(Path('data'), tensor_dataset, bert_tokenizer)
    tensor_dataset = _load_tensor_dataset(Path('data'), partition, bert_tokenizer)
