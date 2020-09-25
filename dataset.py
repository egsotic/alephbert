import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import fasttext_emb as ft

from torch.utils.data import TensorDataset


def xtokenize_token_data(partition, xtokenizer):
    xdata = {}
    for part in partition:
        df = partition[part]
        logging.info(f'processing {part} input {type(xtokenizer).__name__} data')
        raw_df = df[['sent_id', 'token_id', 'token']].drop_duplicates()
        xdf = pd.DataFrame([(row.sent_id, row.token_id, row.token, xt)
                            for index, row in raw_df.iterrows()
                            for xt in xtokenizer.tokenize(row.token)],
                           columns=['sent_id', 'token_id', 'token', 'xtoken'])
        xdata[part] = xdf
    return xdata


def xtokenize_form_data(partition, xtokenizer):
    xdata = {}
    for part in partition:
        df = partition[part]
        raw_df = df[['sent_id', 'token_id', 'form']]
        logging.info(f'processing {part} output {type(xtokenizer).__name__} seg data')
        xdf = pd.DataFrame([(row.sent_id, row.token_id, row.form, xf)
                            for index, row in raw_df.iterrows()
                            for xf in xtokenizer.tokenize(row.form)],
                           columns=['sent_id', 'token_id', 'form', 'xform'])
        xdata[part] = xdf
    return xdata


def save_xemb(xtokenizer):
    xtokens = [xtokenizer.ids_to_tokens[xid] for xid in sorted(xtokenizer.ids_to_tokens)]
    ft.load_embedding_weight_matrix(Path('/Users/Amit/dev/fastText/models/cc.he.300.bin'),
                                    Path(f'data/ft_{type(xtokenizer).__name__}.vec.txt'), list(xtokens))


def save_processed_input_data(data_root_path, input_data, xtokenizer):
    for part in input_data:
        input_df = input_data[part]
        input_data_file = Path(data_root_path) / f'{part}_input_{type(xtokenizer).__name__}_token.csv'
        logging.info(f'saving: {input_data_file}')
        input_df.to_csv(input_data_file)


def save_processed_output_seg_data(data_root_path, output_data, xtokenizer=None):
    for part in output_data:
        output_df = output_data[part]
        if xtokenizer is not None:
            output_data_file = Path(data_root_path) / f'{part}_output_{type(xtokenizer).__name__}_seg.csv'
        else:
            output_data_file = Path(data_root_path) / f'{part}_output_seg.csv'
        logging.info(f'saving: {output_data_file}')
        output_df.to_csv(output_data_file)


def load_processed_input_data(data_root_path, partition, xtokenizer):
    input_data = {}
    for part in partition:
        input_data_file = Path(data_root_path) / f'{part}_input_{type(xtokenizer).__name__}_token.csv'
        logging.info(f'loading: {input_data_file}')
        input_df = pd.read_csv(input_data_file, index_col=0)
        input_data[part] = input_df
    return input_data


def load_processed_output_seg_data(data_root_path, partition, xtokenizer=None):
    output_data = {}
    for part in partition:
        if xtokenizer is not None:
            output_data_file = Path(data_root_path) / f'{part}_output_{type(xtokenizer).__name__}_seg.csv'
        else:
            output_data_file = Path(data_root_path) / f'{part}_output_seg.csv'
        logging.info(f'loading: {output_data_file}')
        output_df = pd.read_csv(output_data_file, index_col=0)
        output_data[part] = output_df
    return output_data


def to_seg_dataset(partition, input_data, output_data, xtokenizer):
    data_samples = {}
    for part in partition:
        logging.info(f'transforming seg dataset: {part} {type(xtokenizer).__name__}')
        input_df = input_data[part]
        output_df = output_data[part]
        input_gb = input_df.groupby(input_df.sent_id)
        output_gb = output_df.groupby(output_df.sent_id)
        inputs = [input_gb.get_group(x).reset_index(drop=True) for x in input_gb.groups]
        outputs = [output_gb.get_group(x).reset_index(drop=True) for x in output_gb.groups]
        max_input_len = max([len(df) for df in inputs])
        max_output_len = max([len(df) for df in outputs])
        samples = []
        for x, y in zip(inputs, outputs):
            sample = _to_seg_sample(x, y, xtokenizer, max_input_len, max_output_len)
            samples.append(sample)
        input_ids = np.array([sample[0] for sample in samples], dtype=np.int)
        input_mask = np.array([sample[1] for sample in samples], dtype=np.int)
        output_ids = np.array([sample[2] for sample in samples], dtype=np.int)
        output_mask = np.array([sample[3] for sample in samples], dtype=np.int)
        data_samples[part] = (input_ids, input_mask, output_ids, output_mask)
    return data_samples


def _to_seg_sample(input_df, output_df, xtokenizer, max_input_len, max_output_len):
    input_xtoken_ids = xtokenizer.convert_tokens_to_ids(input_df.xtoken)
    input_xtoken_ids = [xtokenizer.cls_token_id] + input_xtoken_ids + [xtokenizer.sep_token_id]
    input_xtoken_mask = [1] * len(input_xtoken_ids)
    fill_input_xtoken_mask_len = max_input_len + 2 - len(input_xtoken_ids)
    input_xtoken_ids += [xtokenizer.pad_token_id] * fill_input_xtoken_mask_len
    input_xtoken_mask += [0] * fill_input_xtoken_mask_len

    output_xform_ids = xtokenizer.convert_tokens_to_ids(output_df.xform)
    output_xform_ids = [xtokenizer.cls_token_id] + output_xform_ids + [xtokenizer.sep_token_id]
    output_xform_mask = [1] * len(output_xform_ids)
    fill_output_xform_mask_len = max_output_len - len(output_xform_ids)
    output_xform_ids += [xtokenizer.pad_token_id] * fill_output_xform_mask_len
    output_xform_mask += [0] * fill_output_xform_mask_len
    return input_xtoken_ids, input_xtoken_mask, output_xform_ids, output_xform_mask


def to_seg_tensor_dataset(partition, data_samples, xtokenizer):
    tensor_data = {}
    for part in partition:
        logging.info(f'transforming seg tensor dataset: {part} {type(xtokenizer).__name__}')
        inputs_tensor = torch.tensor(data_samples[part][0], dtype=torch.long)
        inputs_mask_tensor = torch.tensor(data_samples[part][1], dtype=torch.long)
        outputs_tensor = torch.tensor(data_samples[part][2], dtype=torch.long)
        outputs_mask_tensor = torch.tensor(data_samples[part][3], dtype=torch.long)
        tensor_data[part] = TensorDataset(inputs_tensor, inputs_mask_tensor, outputs_tensor, outputs_mask_tensor)
    return tensor_data


def save_seg_tensor_dataset(data_root_path, partition, tensor_data, xtokenizer):
    for part in partition:
        seg_tensor_data_file_path = Path(data_root_path) / f'{part}_{type(xtokenizer).__name__}_seg_tensor_dataset.pt'
        logging.info(f'saving: {seg_tensor_data_file_path}')
        torch.save(tensor_data[part], seg_tensor_data_file_path)


def load_seg_tensor_dataset(data_root_path, partition, xtokenizer):
    tensor_data = {}
    for part in partition:
        seg_tensor_data_file_path = Path(data_root_path) / f'{part}_{type(xtokenizer).__name__}_seg_tensor_dataset.pt'
        logging.info(f'loading: {seg_tensor_data_file_path}')
        tensor_samples = torch.load(seg_tensor_data_file_path)
        tensor_data[part] = tensor_samples
    return tensor_data
