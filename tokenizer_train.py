import logging
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer, BertWordPieceTokenizer

# tokenizer_type = 'bpe-byte'
# tokenizer_type = 'bpe-char'
tokenizer_type = 'wordpiece'


def train_tokenizer(data_file_paths):
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    wordpieces_prefix = None
    if tokenizer_type == 'byte':
        t = ByteLevelBPETokenizer()
    elif tokenizer_type == 'char':
        t = CharBPETokenizer()
    else:
        t = BertWordPieceTokenizer()
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        wordpieces_prefix = "##"
    t.train(
        files=data_file_paths,
        vocab_size=52000,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens,
        limit_alphabet=1000,
        wordpieces_prefix=wordpieces_prefix
    )
    return t


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

paths = [str(x) for x in Path("data").glob("oscar/he_dedup.txt")]
tokenizer = train_tokenizer(paths)
tokenizer.save_model(f'models/tokenizer/{tokenizer_type}')
