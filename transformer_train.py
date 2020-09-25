import logging

from transformers import (
    BertConfig, RobertaConfig, RobertaForMaskedLM, BertForMaskedLM, LineByLineTextDataset,
    DataCollatorForLanguageModeling, TrainingArguments, Trainer, RobertaTokenizerFast, BertTokenizerFast, set_seed
)

# tokenizer_type = 'bpe-byte'
# tokenizer_type = 'bpe-char'
tokenizer_type = 'wordpiece'
# transformer_type = 'roberta'
transformer_type = 'bert'


def get_config():
    if transformer_type == 'roberta':
        return RobertaConfig(
            vocab_size=52000,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )
    return BertConfig(
        vocab_size=52000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )


def get_tokenizer():
    pretrained_tokenizer_path = f'models/tokenizer/{tokenizer_type}'
    if transformer_type == 'roberta':
        return RobertaTokenizerFast.from_pretrained(pretrained_tokenizer_path, max_len=512)
    return BertTokenizerFast.from_pretrained(pretrained_tokenizer_path, max_len=512)


def get_model():
    config = get_config()
    if transformer_type == 'roberta':
        return RobertaForMaskedLM(config=config)
    return BertForMaskedLM(config=config)


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

tokenizer = get_tokenizer()
model = get_model()
print(model.num_parameters())
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data/oscar/he_dedup-train.txt",
    block_size=128,
)
dev_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data/oscar/he_dedup-eval.txt",
    block_size=128,
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
training_args = TrainingArguments(
    output_dir=f'./models/{transformer_type}-{tokenizer_type}-v1',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    save_steps=10000,
    save_total_limit=2,
    # learning_rate=1e-4,
    do_train=True,
    do_eval=True,
    # evaluate_during_training=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    prediction_loss_only=True
)

logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    training_args.local_rank,
    training_args.device,
    training_args.n_gpu,
    bool(training_args.local_rank != -1),
    training_args.fp16,
)
logger.info("Training/evaluation parameters %s", training_args)

set_seed(42)

trainer.train()
trainer.save_model()
# For convenience, we also re-save the tokenizer to the same directory, so that you can share your model easily
tokenizer.save_pretrained(training_args.output_dir)

