import logging
from pathlib import Path
from datasets import load_dataset
from transformers import BertConfig, TrainingArguments, set_seed
# from transformers.data.datasets import LineByLineTextDataset
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.models.bert.modeling_bert import BertForMaskedLM


def get_config():
    return BertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=512,
        num_attention_heads=12,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=1,
    )


def get_tokenizer():
    pretrained_tokenizer_path = Path(f'./experiments/tokenizers/{tokenizer_type}/{tokenizer_type}-{data_source_name}-{vocab_size}')
    logger.info(f'loading tokenizer {pretrained_tokenizer_path}')
    return BertTokenizerFast.from_pretrained(str(pretrained_tokenizer_path), max_len=512)


def get_model(model_path=None):
    if model_path is None:
        config = get_config()
        return BertForMaskedLM(config=config)
    logging.info('Loading pre-trained AlephBERT')
    bert = BertForMaskedLM.from_pretrained(str(model_path))
    bert_tokenizer = BertTokenizerFast.from_pretrained(str(model_path))
    return bert, bert_tokenizer


# def get_train_data1(tokenizer):
#     p = Path('data/raw/oscar') / f'he_dedup-1000.txt'
#     logger.info(f'training data: {p}')
#     return LineByLineTextDataset(
#         tokenizer=tokenizer,
#         file_path=str(p),
#         block_size=128,
#     )


def get_train_data(max_length, min_length=0):
    paths = ['data/raw/oscar/he_dedup.txt']
    logger.info(f'loading training data from: {paths}')
    # ds = load_dataset('text', data_files=[str(p)], cache_dir='/localdata/amitse/.cache')
    ds = load_dataset('text', data_files=paths)

    def tokenize_function(examples):
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
#         batch_encoding = tokenizer(examples["text"], add_special_tokens=True, return_special_tokens_mask=True, truncation=True, max_length=128)
        batch_encoding = tokenizer(examples["text"], add_special_tokens=True, return_special_tokens_mask=False, return_length=True, return_token_type_ids=False, return_attention_mask=False)
#         batch_encoding = tokenizer(examples["text"], add_special_tokens=True, return_special_tokens_mask=False, return_length=True, return_token_type_ids=False, return_attention_mask=False, truncation=True)
        # examples['input_ids'] = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in batch_encoding["input_ids"]]
        return batch_encoding
    return ds.map(
        tokenize_function,
        batched=True,
        num_proc=8,
    ).filter(lambda e: e['length'] > min_length and e['length'] < max_length)


def get_data_collator():
    return DataCollatorForLanguageModeling(tokenizer=tokenizer)


def get_train_args(lr=1e-4):
    p = Path(f'experiments/transformers/bert/distilled/{tokenizer_type}') / f'bert-distilled-{tokenizer_type}-{data_source_name}-{vocab_size}'
    p.mkdir(parents=True, exist_ok=True)
    return TrainingArguments(
        output_dir=str(p),
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=48,
        gradient_accumulation_steps=5,
        # eval_accumulation_steps=1,
        save_total_limit=0,
        save_steps=0,
        learning_rate=lr,
        # fp16=True,
        # logging_steps=10000,
        prediction_loss_only=True,
        dataloader_num_workers=8,
        # local_rank=0,
        # sharded_ddp=True,
    )


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

data_source_name = 'oscar'
tokenizer_type = 'wordpiece'
vocab_size = 52000
num_hidden_layers = 6
training_args = get_train_args()
tokenizer = get_tokenizer()
model = get_model()
data_collator = get_data_collator()
train_dataset = get_train_data(64)

length_series = train_dataset['train'].data[1].to_pandas()
print(f'num samples: {len(length_series)}')
print(f'avg sample length: {length_series.mean(axis=0)}')
print(f'sample length stddev: {length_series.std(axis=0)}')


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
