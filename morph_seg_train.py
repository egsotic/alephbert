import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModelForMaskedLM, BertForMaskedLM, \
    AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoModel
import dataset
from model import MorphSegSeq2SeqModel
import fasttext_emb as ft
from pathlib import Path


# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

# Language Model
# roberta_lm = AutoModelForTokenClassification.from_pretrained("./experiments/transformers/roberta-bpe-byte-v1")
# bert_lm = AutoModelForTokenClassification.from_pretrained("./experiments/transformers/bert-wordpiece-v1")
bert = AutoModel.from_pretrained("./experiments/transformers/bert-wordpiece-v1")
logging.info(f'{type(bert).__name__} loaded')
# roberta_tokenizer = AutoTokenizer.from_pretrained("./experiments/transformers/roberta-bpe-byte-v1")
bert_tokenizer = AutoTokenizer.from_pretrained("./experiments/transformers/bert-wordpiece-v1")
logging.info(f'{type(bert_tokenizer).__name__} loaded')
ft_emb = ft.load_embedding(Path('data/ft_BertTokenizer.vec.txt'))

# Data
partition = ['train', 'dev', 'test']
seg_data_partition = dataset.load_seg_tensor_dataset('data', partition, bert_tokenizer)  # TensorDataset
# seg_tag_data_partition = dataset.load_seg_tag_data('data', partition, bert_tokenizer)  # TensorDataset
# host_data_partition = dataset.load_host_data('data', partition, bert_tokenizer)  # TensorDataset
# host_multitag_data_partition = dataset.load_host_multitag_data('data', partition, bert_tokenizer)  # TensorDataset

# Configuration
train_bach_size = 1
train_dataloader = DataLoader(seg_data_partition['train'], batch_size=train_bach_size, shuffle=False)
dev_dataloader = DataLoader(seg_data_partition['dev'], batch_size=1)
test_dataloader = DataLoader(seg_data_partition['test'], batch_size=1)
epochs = 3
lr = 1e-5
optim_scheduler_warmpup_steps = 1
optim_step_every = 1
optim_max_grad_norm = 5.0
total_train_steps = (len(train_dataloader.dataset) // train_bach_size // optim_step_every // epochs)

# Model
model = MorphSegSeq2SeqModel(ft_emb, bert_tokenizer, bert)
# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = False
parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
# parameters = model.parameters()

# Optimization
adam = AdamW(parameters, lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(adam, num_warmup_steps=optim_scheduler_warmpup_steps,
                                               num_training_steps=total_train_steps)
# optimizer = ModelOptimizer(parameters, optimizer, optim_step_every, optim_max_grad_norm, lr_scheduler)


def process(epoch, phase, model, data, optimizer):
    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        b_in_xtoken_ids = batch[0]
        b_in_xtoken_mask = batch[1]
        b_out_xform_ids = batch[2]
        b_out_xform_mask = batch[3]
        b_scores = model(b_in_xtoken_ids, b_in_xtoken_mask, b_out_xform_ids, b_out_xform_mask)
        b_loss = model.loss(b_scores, b_out_xform_ids, b_out_xform_mask != 0)
        print(f'epoch {epoch}, {phase} step {i + 1}, loss: {b_loss}')
        b_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


device = None
if device is not None:
    model.to(device)
print(model)
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    model.train()
    process(epoch, 'train', model, train_dataloader, adam)
    model.eval()
    with torch.no_grad():
        samples = process(model, dev_dataloader)
