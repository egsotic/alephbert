from data.preprocess_form import *
from data.preprocess_labels import *
from bclm import treebank as tb
from hebrew_root_tokenizer import AlefBERTRootTokenizer


pad, sos, eos, sep = '<pad>', '<s>', '</s>', '<sep>'


if __name__ == '__main__':
    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    tokenizer_type = 'wordpiece'
    # tokenizer_type = 'roots'
    vocab_size = 52000
    # vocab_size = 10000
    corpus_name = 'oscar'
    bert_model_size = 'distilled'
    # bert_version = 'mBERT'
    # bert_version = 'mBERT-cased'
    # bert_version = 'heBERT'
    bert_version = f'bert-{bert_model_size}-{tokenizer_type}-{corpus_name}-{vocab_size}'

    dev_root_path = Path('/Users/Amit/dev')
    tb_root_path = dev_root_path / 'onlplab'

    # tb_root_path = tb_root_path / 'UniversalDependencies'
    tb_root_path = tb_root_path / 'HebrewResources/for_amit_spmrl'
    # tb_root_path = tb_root_path / 'HebrewResources/HebrewTreebank'

    # raw_root_path = Path('data/raw/UD_Hebrew')
    raw_root_path = Path('data/raw/for_amit_spmrl')
    # raw_root_path = Path('data/raw/HebrewTreebank')

    # preprocessed_root_path = Path(f'data/preprocessed/UD_Hebrew/HTB/{bert_version}')
    preprocessed_root_path = Path(f'data/preprocessed/for_amit_spmrl/hebtb/{bert_version}')
    # preprocessed_root_path = Path(f'data/preprocessed/HebrewTreebank/hebtb/{bert_version}')
    preprocessed_root_path.mkdir(parents=True, exist_ok=True)

    if not raw_root_path.exists():
        # raw_partition = tb.ud(raw_root_path, 'HTB', tb_root_path)
        raw_partition = tb.spmrl_ner_conllu(raw_root_path, 'hebtb', tb_root_path)
        # raw_partition = tb.spmrl(raw_root_path, 'hebtb', tb_root_path)
    else:
        # raw_partition = tb.ud(raw_root_path, 'HTB')
        raw_partition = tb.spmrl_ner_conllu(raw_root_path, 'hebtb')
        # raw_partition = tb.spmrl(raw_root_path, 'hebtb')

    bert_root_path = Path(f'./experiments/transformers/bert/{bert_model_size}/{tokenizer_type}/{bert_version}')
    if tokenizer_type == 'roots':
        bert_tokenizer = AlefBERTRootTokenizer(str(bert_root_path / 'vocab.txt'))
    elif bert_version == 'mBERT':
        bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')
    elif bert_version == 'mBERT-cased':
        bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
    elif bert_version == 'heBERT':
        bert_tokenizer = BertTokenizerFast.from_pretrained(f'avichr/{bert_version}')
    else:
        bert_tokenizer = BertTokenizerFast.from_pretrained(str(bert_root_path))

    morph_data = get_morph_data(preprocessed_root_path, raw_partition)
    morph_form_char_data = get_form_char_data(preprocessed_root_path, morph_data, sep=sep, eos=eos)
    token_char_data = get_token_char_data(preprocessed_root_path, morph_data)
    xtoken_df = get_xtoken_data(preprocessed_root_path, morph_data, bert_tokenizer, sos=sos, eos=eos)

    ft_root_path = dev_root_path / 'fastText'
    save_char_vocab(preprocessed_root_path, ft_root_path, raw_partition, pad=pad, sep=sep, sos=sos, eos=eos)
    char_vectors, char_vocab = load_char_vocab(preprocessed_root_path)
    label_vocab = load_label_vocab(preprocessed_root_path, morph_data, pad=pad)

    save_xtoken_data_samples(preprocessed_root_path, xtoken_df, bert_tokenizer, pad=pad)
    save_token_char_data_samples(preprocessed_root_path, token_char_data, char_vocab['char2id'], pad=pad)
    save_form_char_data_samples(preprocessed_root_path, morph_form_char_data, char_vocab['char2id'], pad=pad)
    save_labeled_data_samples(preprocessed_root_path, morph_data, label_vocab['labels2id'], pad=pad)
