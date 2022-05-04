import argparse
import json

from transformers import AutoTokenizer

from bclm import treebank as tb
from constants import PAD, SOS, EOS, SEP
from data.preprocess_form import *
from data.preprocess_labels import *


def main(config):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # config
    bert_tokenizer_name = config['bert_tokenizer_name']
    bert_tokenizer_path = config['bert_tokenizer_path']

    # tokenizer_type = config['tokenizer_type']
    # transformer_type = config['transformer_type']
    # vocab_size = config['vocab_size']
    # corpus_name = config['corpus_name']

    # tokenizer_version = f'{tokenizer_type}-{corpus_name}-{vocab_size}'
    # if transformer_type == 'bert':
    #     transformer_type = f'{transformer_type}-{tokenizer_version}'
    # else:
    #     tokenizer_version = transformer_type

    tb_root_path = Path(config['tb_root_path'])
    raw_root_path = Path(config['raw_root_path'])
    preprocessed_root_path = Path(config['preprocessed_root_path'])
    fasttext_lang = config['fasttext_lang']
    fasttext_model_path = Path(config['fasttext_model_path'])

    preprocessed_root_path = preprocessed_root_path / bert_tokenizer_name
    preprocessed_root_path.mkdir(parents=True, exist_ok=False)

    if not raw_root_path.exists():
        raw_partition = tb.ud(raw_root_path, 'HTB', tb_root_path=tb_root_path)
    else:
        raw_partition = tb.ud(raw_root_path, 'HTB', tb_root_path=None)

    bert_tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer_path)

    morph_data = get_morph_data(preprocessed_root_path, raw_partition)
    morph_form_char_data = get_form_char_data(preprocessed_root_path, morph_data, sep=SEP, eos=EOS)
    token_char_data = get_token_char_data(preprocessed_root_path, morph_data)
    xtoken_df = get_xtoken_data(preprocessed_root_path, morph_data, bert_tokenizer, sos=SOS, eos=EOS)

    save_char_vocab(preprocessed_root_path, fasttext_lang, fasttext_model_path, raw_partition,
                    pad=PAD, sep=SEP, sos=SOS, eos=EOS)
    char_vectors, char_vocab = load_char_vocab(preprocessed_root_path)
    label_vocab = load_label_vocab(preprocessed_root_path, morph_data, pad=PAD)

    save_xtoken_data_samples(preprocessed_root_path, xtoken_df, bert_tokenizer, pad=PAD)
    save_token_char_data_samples(preprocessed_root_path, token_char_data, char_vocab['char2id'], pad=PAD)
    save_form_char_data_samples(preprocessed_root_path, morph_form_char_data, char_vocab['char2id'], pad=PAD)
    save_labeled_data_samples(preprocessed_root_path, morph_data, label_vocab['labels2id'], pad=PAD)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=Path, help='config path')
    args = parser.parse_args()

    config_path = args.config_path

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    logging.info(config)

    main(config)
