import random
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import dataset
from bclm import treebank as tb
from pathlib import Path
from model import MorphSegModel, TokenSeq2SeqMorphSeg
# from sklearn.metrics import accuracy_score, f1_score


# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def get_raw_data(annotated_tensor_data, char_vocab):
    target_data = []
    for i, batch in enumerate(annotated_tensor_data):
        xtoken_data, token_char_data, form_char_data, token_form_char_data = batch
        sent_ids = token_char_data[:, :, 0]
        token_chars = token_char_data[:, :, [1, 3]]
        token_form_chars = token_form_char_data[:, :, 1]
        num_tokens = token_char_data[:, :, 1].max().item()
        max_form_len = token_form_char_data[:, :, 3].unique().item()
        target_chars = pack_padded_labels(token_form_chars, num_tokens, max_form_len, char_vocab)
        target_form_chars = to_token_chars(target_chars, char_vocab)
        target_data.append((sent_ids.numpy(), token_chars.numpy(), target_form_chars.numpy()))
    return dataset.to_raw_data(target_data, char_vocab)


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
index2char = {char2index[c]: c for c in char2index}
char_vocab = {'char2index': char2index, 'index2char': index2char}
logging.info('FastText char embedding vectors and vocab loaded')

# Data
# partition = tb.spmrl('data/raw')
partition = ['train', 'dev', 'test']
dataset_partition = get_tensor_dataset(Path('data'), partition, bert_tokenizer, char_vocab)

# Configuration
train_batch_size = 1
# dataset_partition['train'].tensors = [t[4400:4500] for t in dataset_partition['train'].tensors]
# dataset_partition['dev'].tensors = [t[:100] for t in dataset_partition['dev'].tensors]
train_dataloader = DataLoader(dataset_partition['train'], batch_size=train_batch_size, shuffle=False)
dev_dataloader = DataLoader(dataset_partition['dev'], batch_size=1)
test_dataloader = DataLoader(dataset_partition['test'], batch_size=1)

# Model
char_emb = nn.Embedding.from_pretrained(torch.tensor(char_vectors, dtype=torch.float),
                                        freeze=False, padding_idx=char2index['<pad>'])
num_layers = 2
hidden_size = bert.config.hidden_size // num_layers
enc_dropout = 0.1
dec_dropout = 0.1
out_size = len(char2index)
out_dropout = 0.3
token_morph_seg_model = TokenSeq2SeqMorphSeg(char_emb, hidden_size=hidden_size, num_layers=num_layers,
                                             enc_dropout=enc_dropout, dec_dropout=dec_dropout, out_size=out_size,
                                             out_dropout=out_dropout)
seg_model = MorphSegModel(bert, bert_tokenizer, char_emb, token_morph_seg_model)


def to_token_chars(chars, char_vocab):
    eos_char = char_vocab['char2index']['</s>']
    token_chars_mask = torch.eq(chars, eos_char)
    token_ids = torch.cumsum(token_chars_mask, dim=1) + 1
    token_ids -= token_chars_mask.long()
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


def pack_padded_labels(input_labels, num_tokens, max_form_len, char_vocab):
    input_labels = input_labels[:, :num_tokens * max_form_len]
    is_eos = True
    eos_char = char_vocab['char2index']['</s>']
    output_labels = []
    tokens = 0
    for i, label in enumerate(input_labels[0]):
        label = label.item()
        if i % max_form_len == (max_form_len - 1):
            label = eos_char
        if i % max_form_len == 0:
            is_eos = False
        if not is_eos:
            output_labels.append(label)
            if label == eos_char:
                tokens += 1
        if label == eos_char:
            is_eos = True
    labels = torch.tensor([output_labels], dtype=torch.long)
    return labels


def process(epoch, phase, print_every, model, data, target_raw_data, teacher_forcing_ratio, criterion, optimizer=None,
            scheduler=None, max_grad_norm=None):
    print_loss, total_loss = 0, 0
    print_acc, total_acc = 0, 0
    print_data, total_data = [], []
    sos = torch.tensor([char2index['<s>']], dtype=torch.long, device=device)
    eos = torch.tensor([char2index['</s>']], dtype=torch.long, device=device)
    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        xtoken_data, token_char_data, _, token_form_char_data = batch
        xtokens = xtoken_data[:, :, 1]
        sent_ids = token_char_data[:, :, 0]
        input_token_chars = token_char_data[:, :, 1:]
        target_chars = token_form_char_data[:, :, 1]
        # max_num_tokens = token_form_char_data[:, :, 2].unique().item()
        max_form_len = token_form_char_data[:, :, 3].unique().item()
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            num_tokens, char_scores = model(xtokens, input_token_chars, sos, eos, max_form_len, target_chars)
        else:
            num_tokens, char_scores = model(xtokens, input_token_chars, sos, eos, max_form_len)
        target_chars = target_chars[:, :num_tokens * max_form_len]
        batch_loss = criterion(char_scores[0], target_chars[0])
        print_loss += batch_loss
        if optimizer is not None:
            batch_loss.backward()
            # Gradient clipping
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        # gold_chars = decode(token_form_chars, num_tokens, max_form_len, model.char_vocab)
        # gold_form_chars = to_token_chars(gold_chars, model.char_vocab)
        predicted_chars = model.decode(char_scores)
        predicted_chars = pack_padded_labels(predicted_chars, num_tokens, max_form_len, char_vocab)
        # target_chars = pack_padded_labels(target_chars, num_tokens, max_form_len, char_vocab)
        # acc = accuracy_score(predicted_chars, target_chars)
        # print_acc += acc
        predicted_form_chars = to_token_chars(predicted_chars, char_vocab)
        token_chars = input_token_chars[:, :, [0, 2]]
        print_data.append((sent_ids.cpu().numpy(), token_chars.cpu().numpy(), predicted_form_chars.cpu().numpy()))
        if (i + 1) % print_every == 0:
            print(f'epoch {epoch} {phase}, step {i + 1} loss: {print_loss / len(print_data)}')
            print(f'optimizer.lr = {optimizer.param_groups[0]["lr"]}')
            # print(f'epoch {epoch} {phase}, step {i + 1} acc: {print_acc / len(print_data)}')
            predicted_raw_data = dataset.to_raw_data(print_data, char_vocab)
            print_eval_scores(target_raw_data, predicted_raw_data, i+1)
            print_form_sample(predicted_raw_data)
            total_loss += print_loss
            total_acc += print_acc
            total_data.extend(print_data)
            print_loss, print_acc, print_data = 0, 0, []
    if len(print_data) > 0:
        total_loss += print_loss
        total_data.extend(print_data)
    print(f'epoch {epoch} {phase}, total loss: {total_loss / len(total_data)}')
    # print(f'epoch {epoch} {phase}, total acc: {total_acc / len(total_data)}')
    predicted_raw_data = dataset.to_raw_data(total_data, char_vocab)
    print_eval_scores(target_raw_data, predicted_raw_data, 'total')
    return predicted_raw_data


raw_train_data = get_raw_data(train_dataloader, char_vocab)
raw_dev_data = get_raw_data(dev_dataloader, char_vocab)

device = None
if device is not None:
    seg_model.to(device)
print(seg_model)


epochs = 3
lr = 1e-3
scheduler_warmup_steps = 200
step_every = 1
max_grad_norm = 5.0
num_training_batchs = len(train_dataloader.dataset) // train_batch_size
num_training_steps = num_training_batchs * epochs

# Optimization
# freeze bert
for param in bert.parameters():
    param.requires_grad = False
parameters = list(filter(lambda p: p.requires_grad, seg_model.parameters()))
# parameters = seg_model.parameters()
adam = AdamW(parameters, lr=lr)
loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=char_vocab['char2index']['<pad>'])

lr_scheduler = get_linear_schedule_with_warmup(adam, num_warmup_steps=scheduler_warmup_steps,
                                               num_training_steps=num_training_steps)
# optimizer = ModelOptimizer(parameters, optimizer, optim_step_every, optim_max_grad_norm, lr_scheduler)

teacher_forcing_ratio = 1.0
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    seg_model.train()
    process(epoch, 'train', 10, seg_model, train_dataloader, raw_train_data, teacher_forcing_ratio, loss_fct, adam,
            lr_scheduler, max_grad_norm)
    seg_model.eval()
    with torch.no_grad():
        samples = process(epoch, 'dev', 10, seg_model, dev_dataloader, raw_dev_data, 0.0, loss_fct)
