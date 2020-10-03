import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import AutoTokenizer, AutoModel
# get_linear_schedule_with_warmup
# AutoModelForMaskedLM, BertForMaskedLM, \
# AutoModelForSeq2SeqLM, AutoModelForTokenClassification
import dataset
from model import MorphSegSeq2SeqModel, TokenCharEmbedding, Attention, AttnDecoder
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

# Data
partition = ['train', 'dev', 'test']
dataset_partition = dataset.load_tensor_dataset(Path('data'), partition, bert_tokenizer)  # TensorDataset
# seg_tag_data_partition = dataset.load_seg_tag_data('data', partition, bert_tokenizer)  # TensorDataset
# host_data_partition = dataset.load_host_data('data', partition, bert_tokenizer)  # TensorDataset
# host_multitag_data_partition = dataset.load_host_multitag_data('data', partition, bert_tokenizer)  # TensorDataset

# Configuration
train_batch_size = 1
train_dataloader = DataLoader(dataset_partition['train'], batch_size=train_batch_size, shuffle=False)
dev_dataloader = DataLoader(dataset_partition['dev'], batch_size=1)
test_dataloader = DataLoader(dataset_partition['test'], batch_size=1)
epochs = 1
lr = 1e-3
optim_scheduler_warmup_steps = 1
optim_step_every = 1
optim_max_grad_norm = 5.0
total_train_steps = (len(train_dataloader.dataset) // train_batch_size // optim_step_every // epochs)

# Model
ft_char_vectors, char2index = dataset.load_form_char_emb()
index2char = {char2index[c]: c for c in char2index}
# emb = TokenCharEmbedding(nn.Embedding.from_pretrained(ft_char_vectors, freeze=False, padding_idx=char2index['<pad>']))
emb = nn.Embedding.from_pretrained(torch.tensor(ft_char_vectors, dtype=torch.float), freeze=False, padding_idx=char2index['<pad>'])

hidden_size = emb.embedding_dim + bert.config.hidden_size
# vocab_size = bert_tokenizer.vocab_size
vocab_size = len(char2index)
max_seq_len = bert.config.max_position_embeddings
decoder_hidden_size = bert.config.hidden_size
attn = Attention(hidden_size, vocab_size, max_seq_len)
attn_decoder = AttnDecoder(emb, hidden_size, decoder_hidden_size, vocab_size, attn)
seq2seq = MorphSegSeq2SeqModel(bert_tokenizer, bert, char2index, attn_decoder)
# freeze all the parameters
# for param in bert.parameters():
#     param.requires_grad = False
# parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
parameters = seq2seq.parameters()

# Optimization
adam = AdamW(parameters, lr=lr)
# lr_scheduler = get_linear_schedule_with_warmup(adam, num_warmup_steps=optim_scheduler_warmup_steps,
#                                                num_training_steps=total_train_steps)
# optimizer = ModelOptimizer(parameters, optimizer, optim_step_every, optim_max_grad_norm, lr_scheduler)


def process(epoch, phase, model, data, optimizer=None):
    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        b_in_xtoken_ids = batch[0]
        b_in_xtoken_mask = batch[1]
        b_out_char_ids = batch[2]
        b_out_char_mask = batch[3]
        b_scores = model(b_in_xtoken_ids, b_in_xtoken_mask, b_out_char_ids)
        b_loss = model.loss(b_scores, b_out_char_ids, b_out_char_mask)
        print(f'epoch {epoch}, {phase} step {i + 1}, loss: {b_loss}')
        if optimizer is not None:
            b_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        decoded_char_ids = model.decode(b_scores)
        decoded_chars = [index2char[char_id.item()] for char_id in decoded_char_ids.squeeze(0)]
        print(decoded_chars)


device = None
if device is not None:
    seq2seq.to(device)
print(seq2seq)
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    seq2seq.train()
    process(epoch, 'train', seq2seq, train_dataloader, adam)
    seq2seq.eval()
    with torch.no_grad():
        samples = process(epoch, 'dev', seq2seq, dev_dataloader)
