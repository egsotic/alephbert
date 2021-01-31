import logging
from pathlib import Path
from datasets import load_dataset
from transformers import BertConfig, TrainingArguments, set_seed
from transformers.data.datasets import LineByLineTextDataset
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.models.bert.modeling_bert import BertForMaskedLM


def get_config(vocab_size, num_hidden_layers=6):
    return BertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=1,
    )


def get_tokenizer(vocab_size):
    pretrained_tokenizer_path = Path('experiments/tokenizers') / f'{tokenizer_type}-{vocab_size}'
    logger.info(f'loading tokenizer from {pretrained_tokenizer_path}')
    return BertTokenizerFast.from_pretrained(str(pretrained_tokenizer_path), max_len=514)


def get_model(vocab_size):
    config = get_config(vocab_size)
    return BertForMaskedLM(config=config)


def get_train_data1(tokenizer):
    p = Path('data/raw/oscar') / f'he_dedup-1000.txt'
    logger.info(f'training data: {p}')
    return LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=str(p),
        block_size=128,
    )


def get_train_data(max_length):
    p = Path('data/raw/oscar/he_dedup-1000.txt')
    logger.info(f'loading training data from: {p}')
    ds = load_dataset('text', data_files=[str(p)])

    def tokenize_function(examples):
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        batch_encoding = tokenizer(examples["text"], add_special_tokens=True, return_special_tokens_mask=False,
                                   return_length=True, return_token_type_ids=False, return_attention_mask=False)
        return batch_encoding

    return ds.map(
        tokenize_function,
        batched=True,
        # remove_columns=["text"],
        # num_proc=2,
    ).filter(lambda e: e['length'] <= max_length)


def get_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer)


def get_train_args(epochs=1, lr=1e-4):
    p = Path('experiments/transformers') / f'bert-{tokenizer_type}-{vocab_size}'
    # p.mkdir(parents=True, exist_ok=True)
    return TrainingArguments(
        output_dir=str(p),
        overwrite_output_dir=True,
        # num_train_epochs=epochs,
        # per_device_train_batch_size=32,
        # save_steps=10000,
        # save_total_limit=2,
        # learning_rate=lr,
        # do_train=True,
        # do_eval=True,
        # evaluation_strategy='steps',
        # fp16=True,
        # logging_steps=10000
    )


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

tokenizer_type = 'wordpiece'
vocab_size = 52000
training_args = get_train_args()
tokenizer = get_tokenizer(vocab_size)
model = get_model(vocab_size)
data_collator = get_data_collator(tokenizer)
train_dataset = get_train_data(128)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset['train'],
)

set_seed(42)
trainer.train()
trainer.save_model()
# For convenience, we also re-save the tokenizer to the same directory, so that you can share your model easily
tokenizer.save_pretrained(training_args.output_dir)
