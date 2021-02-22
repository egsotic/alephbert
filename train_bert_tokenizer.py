import logging
from pathlib import Path
from tokenizers import BertWordPieceTokenizer


def train_tokenizer(data_file_paths, vocab_size):
    t = BertWordPieceTokenizer()
    t.train(
        files=data_file_paths,
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        limit_alphabet=1000,
    )
    return t


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

vocab_size = 52000
# paths = [str(x) for x in Path("./data/raw").glob("**/*.txt")]
paths = ['data/raw/oscar/he_dedup.txt']
tokenizer = train_tokenizer(paths, vocab_size)
tokenizer_folder_path = Path(f'./experiments/tokenizers/wordpiece/wordpiece-oscar-{vocab_size}')
tokenizer.save_model(str(tokenizer_folder_path))
