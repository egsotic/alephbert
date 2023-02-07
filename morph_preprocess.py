import argparse
import json

from bclm import treebank as tb
from constants import PAD, SOS, EOS, SEP
from data.preprocess_form import *
from data.preprocess_labels import *
from utils import get_ud_preprocessed_dir_path, get_tokenizer


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
    oracle_tokenizer = config.get('oracle_tokenizer', False)
    tokenizer_type = config.get('tokenizer_type', 'auto')

    if oracle_tokenizer:
        bert_tokenizer_name = f'oracle_{bert_tokenizer_name}'

    # tokenizer_type = config['tokenizer_type']
    # transformer_type = config['transformer_type']
    # vocab_size = config['vocab_size']
    # corpus_name = config['corpus_name']

    # tokenizer_version = f'{tokenizer_type}-{corpus_name}-{vocab_size}'
    # if transformer_type == 'bert':
    #     transformer_type = f'{transformer_type}-{tokenizer_version}'
    # else:
    #     tokenizer_version = transformer_type

    # UD dir name format: UD_{lang}-{tb_name}
    # UD file name format: {la_name}_{tb_name}-ud-{partition_type}.conllu
    lang = config['lang']
    tb_name = config['tb_name']
    la_name = config['la_name']
    tb_root_path = Path(config['tb_root_path'])
    raw_root_path = Path(config['raw_root_path'])
    preprocessed_root_path = Path(config['preprocessed_root_path'])
    fasttext_lang = config['fasttext_lang']
    fasttext_model_path = Path(config['fasttext_model_path'])
    overwrite_existing = config.get('overwrite_existing', False)

    preprocessed_dir_path = get_ud_preprocessed_dir_path(preprocessed_root_path, lang, tb_name, bert_tokenizer_name)
    preprocessed_dir_path.mkdir(parents=True, exist_ok=overwrite_existing)

    raw_partition = tb.ud(raw_root_path,
                          tb_root_path=tb_root_path,
                          lang=lang,
                          tb_name=tb_name,
                          la_name=la_name,
                          overwrite_existing=overwrite_existing)

    # tokenizer
    bert_tokenizer = get_tokenizer(tokenizer_type, bert_tokenizer_path)

    # oracle
    if oracle_tokenizer:
        make_tokenizer_oracle(bert_tokenizer)

    morph_data = get_morph_data(preprocessed_dir_path, raw_partition)
    morph_form_char_data = get_form_char_data(preprocessed_dir_path, morph_data, sep=SEP, eos=EOS)
    token_char_data = get_token_char_data(preprocessed_dir_path, morph_data)
    xtoken_df = get_xtoken_data(preprocessed_dir_path, morph_data, bert_tokenizer, sos=SOS, eos=EOS)

    save_char_vocab(preprocessed_dir_path, fasttext_lang, fasttext_model_path, raw_partition,
                    pad=PAD, sep=SEP, sos=SOS, eos=EOS)
    char_vectors, char_vocab = load_char_vocab(preprocessed_dir_path)
    label_vocab = load_label_vocab(preprocessed_dir_path, morph_data, pad=PAD)

    save_xtoken_data_samples(preprocessed_dir_path, xtoken_df, bert_tokenizer, pad=PAD)
    save_token_char_data_samples(preprocessed_dir_path, token_char_data, char_vocab['char2id'], pad=PAD)
    save_form_char_data_samples(preprocessed_dir_path, morph_form_char_data, char_vocab['char2id'], pad=PAD)
    save_labeled_data_samples(preprocessed_dir_path, morph_data, label_vocab['labels2id'], pad=PAD)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=Path, help='config path')
    args = parser.parse_args()

    config_path = args.config_path

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    logging.info(config)

    main(config)
