import random
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
from bclm import treebank as tb
from model2 import Model
from pathlib import Path


# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def get_raw_data(annotated_tensor_data, model):
    seg_data = []
    for i, batch in enumerate(annotated_tensor_data):
        sent_ids = batch[0][:, :, 0]
        token_chars = batch[1][:, :, 1:]
        form_chars = batch[2][:, :, 1]
        form_chars = to_token_chars(form_chars, model.char_vocab)
        seg_data.append((sent_ids.numpy(), token_chars.numpy(), form_chars.numpy()))
    return dataset.to_raw_data(seg_data, model.char_vocab)


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
            xtoken_data, token_char_data, form_char_data, token_form_char_data = data[part]
            xtoken_data = torch.tensor(xtoken_data, dtype=torch.long)
            token_char_data = torch.tensor(token_char_data, dtype=torch.long)
            form_char_data = torch.tensor(form_char_data, dtype=torch.long)
            token_form_char_data = torch.tensor(token_form_char_data, dtype=torch.long)
            tensor_dataset = TensorDataset(xtoken_data, token_char_data, form_char_data, token_form_char_data)
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
lr = 1e-3
optim_scheduler_warmup_steps = 1
optim_step_every = 1
optim_max_grad_norm = 5.0

# Model
index2char = {char2index[c]: c for c in char2index}
char_emb = nn.Embedding.from_pretrained(torch.tensor(char_vectors, dtype=torch.float),
                                        freeze=False, padding_idx=char2index['<pad>'])
seg_model = Model(bert, bert_tokenizer, char_emb, {'char2index': char2index, 'index2char': index2char}, enc_num_layers=1, enc_dropout=0.0, dec_num_layers=1, dec_dropout=0.0)
# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = False
parameters = list(filter(lambda p: p.requires_grad, seg_model.parameters()))
# parameters = seg_model.parameters()

# Optimization
adam = AdamW(parameters, lr=lr)
# lr_scheduler = get_linear_schedule_with_warmup(adam, num_warmup_steps=optim_scheduler_warmup_steps,
#                                                num_training_steps=total_train_steps)
# optimizer = ModelOptimizer(parameters, optimizer, optim_step_every, optim_max_grad_norm, lr_scheduler)


def to_token_chars(chars, char_vocab):
    eos_char = char_vocab['char2index']['</s>']
    token_chars_mask = torch.eq(chars, eos_char)
    token_ids = torch.cumsum(token_chars_mask, dim=1) + 1
    token_ids -= token_chars_mask.long()
    num_tokens = token_ids[:, -1:]
    token_pad_mask = torch.eq(token_ids, num_tokens)
    token_ids[token_pad_mask] = 0
    return torch.stack([token_ids, chars], dim=2)


def to_form_list(df):
    return [form for form in df['form'].tolist()]


def print_eval_scores(truth_df, decoded_df, step):
    if len(truth_df['sent_id'].unique()) != len(decoded_df['sent_id'].unique()):
        truth_df = truth_df.loc[truth_df['sent_id'].isin(decoded_df.sent_id.unique().tolist())]
    aligned_scores = tb.seg_eval(truth_df, decoded_df, False)
    mset_scores = tb.seg_eval(truth_df, decoded_df, True)
    print(f'eval {step} aligned: [P: {aligned_scores[0]}, R: {aligned_scores[1]}, F: {aligned_scores[2]}]')
    print(f'eval {step} mset   : [P: {mset_scores[0]}, R: {mset_scores[1]}, F: {mset_scores[2]}]')


def print_form_sample(df):
    gb = list(df.groupby('sent_id'))
    forms = to_form_list(gb[-1][1])
    print(' '.join(forms))


def decode(labels, num_tokens, max_form_len, char_vocab):
    labels = labels[:, :num_tokens * max_form_len]
    mask = []
    is_eos = False
    eos_char = char_vocab['char2index']['</s>']
    for i, label in enumerate(labels[0]):
        if i % max_form_len == 0:
            is_eos = False
        mask.append(is_eos)
        if labels[:, i] == eos_char:
            is_eos = True
    labels = labels[:, [not elem for elem in mask]]
    return labels


def process(epoch, phase, print_every, model, data, raw_truth_data, teacher_forcing_ratio, optimizer=None):
    print_loss, total_loss = 0, 0
    print_data, total_data = [], []
    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        input_sent_ids = batch[0][:, :, 0]
        input_xtokens = batch[0][:, :, 1]
        input_mask = input_xtokens != model.xtokenizer.pad_token_id
        input_token_chars = batch[1][:, :, 1:]
        # output_sent_chars = batch[2][:, :, 1]
        output_token_chars = batch[3][:, :, 1]
        max_num_tokens = batch[3][:, :, 2].unique().item()
        max_form_len = batch[3][:, :, 3].unique().item()
        # output_mask = output_token_chars != model.char_vocab['char2index']['<pad>']
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        model_scores = model(input_xtokens, input_mask, input_token_chars, output_token_chars, max_num_tokens,
                             max_form_len, use_teacher_forcing)
        num_tokens = input_token_chars[:, : 1].max()
        decoded_chars = model.decode(model_scores)
        decoded_chars = decode(decoded_chars, num_tokens, max_form_len, model.char_vocab)
        decoded_form_chars = to_token_chars(decoded_chars, model.char_vocab)
        print_data.append((input_sent_ids.cpu().numpy(), input_token_chars.cpu().numpy(),
                           decoded_form_chars.cpu().numpy()))
        batch_loss = model.loss(model_scores, output_token_chars)
        print_loss += batch_loss
        if optimizer is not None:
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (i + 1) % print_every == 0:
            print(f'epoch {epoch} {phase}, step {i + 1} loss: {print_loss / len(print_data)}')
            raw_decoded_data = dataset.to_raw_data(print_data, model.char_vocab)
            print_eval_scores(raw_truth_data, raw_decoded_data, i+1)
            print_form_sample(raw_decoded_data)
            total_loss += print_loss
            total_data.extend(print_data)
            print_loss, print_data = 0, []
    if len(print_data) > 0:
        total_loss += print_loss
        total_data.extend(print_data)
    print(f'epoch {epoch} {phase}, total loss: {total_loss / len(total_data)}')
    raw_decoded_data = dataset.to_raw_data(total_data, model.char_vocab)
    print_eval_scores(raw_truth_data, raw_decoded_data, 'total')
    return raw_decoded_data


raw_train_data = get_raw_data(train_dataloader, seg_model)
raw_dev_data = get_raw_data(dev_dataloader, seg_model)

device = None
if device is not None:
    seg_model.to(device)
print(seg_model)
teacher_forcing_ratio = 1.0
epochs = 1
total_train_steps = (len(train_dataloader.dataset) // train_batch_size // optim_step_every // epochs)
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    seg_model.train()
    process(epoch, 'train', 10, seg_model, train_dataloader, raw_train_data, teacher_forcing_ratio, adam)
    seg_model.eval()
    with torch.no_grad():
        samples = process(epoch, 'dev', 10, seg_model, dev_dataloader, raw_dev_data, 0.0)
