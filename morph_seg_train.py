import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from transformers import AutoTokenizer, AutoModel
# get_linear_schedule_with_warmup
# AutoModelForMaskedLM, BertForMaskedLM, \
# AutoModelForSeq2SeqLM, AutoModelForTokenClassification
import dataset
from model2 import Model
from pathlib import Path


# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def get_tensor_dataset(data_path, partition, xtokenizer, char_vocab):
    tensor_data = {}
    tensor_data_files = [data_path / f'{part}_{type(xtokenizer).__name__}_data.pt' for part in partition]
    if all([f.exists() for f in tensor_data_files]):
        for part in partition:
            tensor_data_file_path = data_path / f'{part}_{type(xtokenizer).__name__}_data.pt'
            logging.info(f'loading tensor data from file {tensor_data_file_path}')
            tensor_dataset = torch.load(tensor_data_file_path)
            tensor_data[part] = tensor_dataset
    else:
        data = dataset.load_data(partition, xtokenizer, char_vocab)
        for part in data:
            xtoken_data, token_char_data, form_char_data = data[part]
            xtoken_data = torch.tensor(xtoken_data, dtype=torch.long)
            token_char_data = torch.tensor(token_char_data, dtype=torch.long)
            form_char_data = torch.tensor(form_char_data, dtype=torch.long)
            tensor_dataset = TensorDataset(xtoken_data, token_char_data, form_char_data)
            tensor_data_file_path = data_path / f'{part}_{type(xtokenizer).__name__}_data.pt'
            logging.info(f'saving tensor data to file {tensor_data_file_path}')
            torch.save(tensor_dataset, tensor_data_file_path)
            tensor_data[part] = tensor_dataset
    return tensor_data


# BERT Language Model
bert_folder_path = Path('./experiments/transformers/bert-wordpiece-v1')
logging.info(f'BERT folder path: {str(bert_folder_path)}')
bert = AutoModel.from_pretrained(str(bert_folder_path))
bert_tokenizer = AutoTokenizer.from_pretrained(str(bert_folder_path))
logging.info('BERT model and tokenizer loaded')

# FastText
char_vectors, char2index = dataset.load_emb('char')
logging.info('FastText char embedding vectors and vocab loaded')

# Data
# partition = tb.spmrl('data/raw')
partition = ['train', 'dev', 'test']
dataset_partition = get_tensor_dataset(Path('data'), partition, bert_tokenizer, char2index)

# Configuration
train_batch_size = 1
train_dataloader = DataLoader(dataset_partition['train'], batch_size=train_batch_size, shuffle=False)
dev_dataloader = DataLoader(dataset_partition['dev'], batch_size=1)
test_dataloader = DataLoader(dataset_partition['test'], batch_size=1)
epochs = 1
lr = 1e-5
optim_scheduler_warmup_steps = 1
optim_step_every = 1
optim_max_grad_norm = 5.0
total_train_steps = (len(train_dataloader.dataset) // train_batch_size // optim_step_every // epochs)

# Model
index2char = {char2index[c]: c for c in char2index}
char_emb = nn.Embedding.from_pretrained(torch.tensor(char_vectors, dtype=torch.float),
                                        freeze=False, padding_idx=char2index['<pad>'])
seg_model = Model(bert, char_emb, char2index, enc_num_layers=1, enc_dropout=0.0, dec_num_layers=1, dec_dropout=0.0)
# freeze all the parameters
# for param in bert.parameters():
#     param.requires_grad = False
# parameters = list(filter(lambda p: p.requires_grad, seg_model.parameters()))
parameters = seg_model.parameters()

# Optimization
adam = AdamW(parameters, lr=lr)
# lr_scheduler = get_linear_schedule_with_warmup(adam, num_warmup_steps=optim_scheduler_warmup_steps,
#                                                num_training_steps=total_train_steps)
# optimizer = ModelOptimizer(parameters, optimizer, optim_step_every, optim_max_grad_norm, lr_scheduler)


def process(epoch, phase, model, data, optimizer=None):
    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        input_xtokens = batch[0][:, :, 1]
        input_mask = input_xtokens != bert_tokenizer.pad_token_id
        token_chars = batch[1][:, :, 1:]
        output_chars = batch[2][:, :, 1]
        output_mask = output_chars != model.char_vocab['<pad>']
        b_scores = model(input_xtokens, input_mask, token_chars, output_chars)
        b_loss = model.loss(b_scores, output_chars, output_mask)
        print(f'epoch {epoch}, {phase} step {i + 1}, loss: {b_loss}')
        if optimizer is not None:
            b_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        decoded_char_ids = model.decode(b_scores)
        decoded_chars = [index2char[char_id.item()] for char_id in decoded_char_ids.squeeze(0)]
        decoded_chars = [' ' if c == '<sep>' else c for c in decoded_chars]
        print(''.join(decoded_chars))


device = None
if device is not None:
    seg_model.to(device)
print(seg_model)
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    seg_model.train()
    process(epoch, 'train', seg_model, train_dataloader, adam)
    seg_model.eval()
    with torch.no_grad():
        samples = process(epoch, 'dev', seg_model, dev_dataloader)
