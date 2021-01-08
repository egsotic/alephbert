import logging
from pathlib import Path
from transformers import (
    BertConfig, RobertaConfig, RobertaForMaskedLM, BertForMaskedLM, LineByLineTextDataset, TextDataset,
    DataCollatorForLanguageModeling, TrainingArguments, Trainer, RobertaTokenizerFast, BertTokenizerFast, set_seed
)
from datasets import load_dataset


def get_config(vocab_size):
    if transformer_type == 'roberta':
        return RobertaConfig(
            vocab_size=vocab_size,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )
    return BertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )


def get_tokenizer(vocab_size):
    pretrained_tokenizer_path = Path('experiments/tokenizers') / f'{tokenizer_type}-{vocab_size}'
    logger.info(f'loading {tokenizer_type}-{vocab_size} tokenizer from {pretrained_tokenizer_path}')
    if transformer_type == 'roberta':
        return RobertaTokenizerFast.from_pretrained(str(pretrained_tokenizer_path), max_len=512)
    return BertTokenizerFast.from_pretrained(str(pretrained_tokenizer_path), max_len=512)


def get_model(vocab_size):
    config = get_config(vocab_size)
    if transformer_type == 'roberta':
        return RobertaForMaskedLM(config=config)
    return BertForMaskedLM(config=config)


def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length')


def get_train_data1(tokenizer, epoch):
    p = Path('data/raw/oscar') / f'he_dedup-train-{(epoch % 2) + 1}.txt'
    logger.info(f'{transformer_type} training data: {p}')
    return LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=str(p),
        block_size=128,
    )


def get_train_data2(tokenizer):
    p = Path('data/raw/oscar/he_dedup-1000.txt')
    logger.info(f'{transformer_type} training data: {p}')
    return TextDataset(
        tokenizer=tokenizer,
        file_path=str(p),
        block_size=128,
    )


def get_train_data3():
    p = Path('data/raw/oscar/he_dedup-1000.txt')
    logger.info(f'loading training data from: {p}')
    ds = load_dataset('text', data_files=[str(p)])
    padding = "max_length"
    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return tokenizer(
            examples["text"],
            # padding=padding,
            truncation=True,
            # max_length=512,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

    return ds.map(
        tokenize_function,
        batched=True,
        num_proc=2,
        remove_columns=["text"],
        load_from_cache_file=True,
    )


# def get_dev_data():
#     return LineByLineTextDataset(
#         tokenizer=tokenizer,
#         file_path='./data/raw/oscar/he_dedup-eval.txt',
#         block_size=128,
#     )


def get_data_collator():
    return DataCollatorForLanguageModeling(tokenizer=tokenizer)


def get_train_args(epochs=1, lr=1e-4):
    p = Path('experiments/transformers') / f'{transformer_type}-{tokenizer_type}-{vocab_size}'
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


# tokenizer_type = 'bpe-byte'
# tokenizer_type = 'bpe-char'
tokenizer_type = 'wordpiece'

# transformer_type = 'roberta'
transformer_type = 'bert'

# vocab_size = 52000
vocab_size = 52000
training_args = get_train_args()
tokenizer = get_tokenizer(vocab_size)
model = get_model(vocab_size)
data_collator = get_data_collator()
# dev_dataset = get_dev_data()
# train_dataset = get_train_data2(tokenizer)
train_dataset = get_train_data3()

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset['train'],
    # eval_dataset=dev_dataset,
    # prediction_loss_only=True
)

# logger.warning(
#     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
#     training_args.local_rank,
#     training_args.device,
#     training_args.n_gpu,
#     bool(training_args.local_rank != -1),
#     training_args.fp16,
# )
# logger.info("Training/evaluation parameters %s", training_args)
set_seed(42)
trainer.train()
trainer.save_model()
# For convenience, we also re-save the tokenizer to the same directory, so that you can share your model easily
tokenizer.save_pretrained(training_args.output_dir)
