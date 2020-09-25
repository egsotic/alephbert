# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from transformers import AutoTokenizer
import logging
from bclm import treebank as tb
import dataset


def load_tensor_data(roberta_tokenizer, bert_tokenizer):
    partition = {'train': None, 'dev': None, 'test': None}
    dataset.load_seg_tensor_dataset('data', partition, bert_tokenizer)
    dataset.load_seg_tensor_dataset('data', partition, roberta_tokenizer)


def save_tensor_data(roberta_tokenizer, bert_tokenizer):
    partition = {'train': None, 'dev': None, 'test': None}
    roberta_input_data = dataset.load_processed_input_data('data/processed/hebtb', partition, roberta_tokenizer)
    bert_input_data = dataset.load_processed_input_data('data/processed/hebtb', partition, bert_tokenizer)
    roberta_seg_output_data = dataset.load_processed_output_seg_data('data/processed/hebtb', partition, roberta_tokenizer)
    bert_seg_output_data = dataset.load_processed_output_seg_data('data/processed/hebtb', partition, bert_tokenizer)
    seg_dataset = dataset.to_seg_dataset(partition, bert_input_data, bert_seg_output_data, bert_tokenizer)
    seg_tensor_dataset = dataset.to_seg_tensor_dataset(partition, seg_dataset, bert_tokenizer)
    dataset.save_seg_tensor_dataset('data', partition, seg_tensor_dataset, bert_tokenizer)
    seg_dataset = dataset.to_seg_dataset(partition, roberta_input_data, roberta_seg_output_data, roberta_tokenizer)
    seg_tensor_dataset = dataset.to_seg_tensor_dataset(partition, seg_dataset, roberta_tokenizer)
    dataset.save_seg_tensor_dataset('data', partition, seg_tensor_dataset, roberta_tokenizer)


def save_processed_data(roberta_tokenizer, bert_tokenizer):
    partition = tb.spmrl('data/raw')
    roberta_input_data = dataset.xtokenize_token_data(partition, roberta_tokenizer)
    dataset.save_processed_input_data('data/processed/hebtb', roberta_input_data, roberta_tokenizer)
    bert_input_data = dataset.xtokenize_token_data(partition, bert_tokenizer)
    dataset.save_processed_input_data('data/processed/hebtb', bert_input_data, bert_tokenizer)
    roberta_seg_output_data = dataset.xtokenize_form_data(partition, roberta_tokenizer)
    dataset.save_processed_output_seg_data('data/processed/hebtb', roberta_seg_output_data, roberta_tokenizer)
    bert_seg_output_data = dataset.xtokenize_form_data(partition, bert_tokenizer)
    dataset.save_processed_output_seg_data('data/processed/hebtb', bert_seg_output_data, bert_tokenizer)


if __name__ == '__main__':
    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    roberta_tokenizer = AutoTokenizer.from_pretrained("./experiments/transformers/roberta-bpe-byte-v1")
    logging.info(f'{type(roberta_tokenizer).__name__} loaded')
    bert_tokenizer = AutoTokenizer.from_pretrained("./experiments/transformers/bert-wordpiece-v1")
    logging.info(f'{type(bert_tokenizer).__name__} loaded')
    # save_processed_data(roberta_tokenizer, bert_tokenizer)
    # save_tensor_data(roberta_tokenizer, bert_tokenizer)
    # load_tensor_data(roberta_tokenizer, bert_tokenizer)
    dataset.save_xemb(bert_tokenizer)
