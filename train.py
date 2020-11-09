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
from model_tag import MorphTagModel, TokenSeq2SeqMorphTagger
# from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import itertools


# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

# Language Model
bert_folder_path = Path('./experiments/transformers/bert-wordpiece-v1')
logging.info(f'BERT folder path: {str(bert_folder_path)}')
bert = AutoModel.from_pretrained(str(bert_folder_path))
bert_tokenizer = AutoTokenizer.from_pretrained(str(bert_folder_path))
logging.info('BERT model and tokenizer loaded')

# Data
# partition = tb.spmrl('data/raw')
partition = ['train', 'dev', 'test']
logging.info('Loading vocabularies')
char_vectors, char_vocab, tag_vocab, feats_vocab = dataset.load_vocab(partition, bert_tokenizer)


def load_data(partition, xtokenizer, char_vocab, tag_vocab, feats_vocab):
    logging.info('Loading data')
    tensor_data = {}
    data = dataset.load_data(partition, xtokenizer, char_vocab, tag_vocab, feats_vocab)
    for part in partition:
        token_data, token_char_data, form_token_char_data, lemma_token_char_data, morph_token_data = data[part]
        token_data = torch.tensor(token_data, dtype=torch.long)
        token_char_data = torch.tensor(token_char_data, dtype=torch.long)
        token_form_char_data = torch.tensor(form_token_char_data, dtype=torch.long)
        token_lemma_char_data = torch.tensor(lemma_token_char_data, dtype=torch.long)
        token_morph_data = torch.tensor(morph_token_data, dtype=torch.long)
        ds = TensorDataset(token_data, token_char_data, token_form_char_data, token_lemma_char_data, token_morph_data)
        tensor_data[part] = ds
    return tensor_data


def save_tensors(tensors_data, xtokenizer, data_path):
    for part in tensors_data:
        data = tensors_data[part]
        tensor_data_file_path = data_path / f'{part}_{type(xtokenizer).__name__}_data.pt'
        logging.info(f'Saving tensor dataset to file {tensor_data_file_path}')
        torch.save(data, tensor_data_file_path)


def load_tensors(partition, xtokenizer, data_path):
    data = {}
    for part in partition:
        tensor_data_file_path = data_path / f'{part}_{type(xtokenizer).__name__}_data.pt'
        logging.info(f'Loading tensor dataset from file {tensor_data_file_path}')
        tensor_dataset = torch.load(tensor_data_file_path)
        data[part] = tensor_dataset
    return data


data_path = Path('data')
tensor_data_files = [data_path / f'{part}_{type(bert_tokenizer).__name__}_data.pt' for part in partition]
if all([f.exists() for f in tensor_data_files]):
    dataset_partition = load_tensors(partition, bert_tokenizer, data_path)
else:
    dataset_partition = load_data(partition, bert_tokenizer, char_vocab, tag_vocab, feats_vocab)
    save_tensors(dataset_partition, bert_tokenizer, data_path)

train_batch_size = 1
# dataset_partition['train'].tensors = [t[:100] for t in dataset_partition['train'].tensors]
# dataset_partition['dev'].tensors = [t[:100] for t in dataset_partition['dev'].tensors]
# dataset_partition['test'].tensors = [t[:100] for t in dataset_partition['test'].tensors]
train_dataloader = DataLoader(dataset_partition['train'], batch_size=train_batch_size, shuffle=False)
dev_dataloader = DataLoader(dataset_partition['dev'], batch_size=1)
test_dataloader = DataLoader(dataset_partition['test'], batch_size=1)


def to_raw_data(data, char_vocab, tag_vocab, feats_vocab):
    raw_data = []
    for batch in data:
        token_data, token_char_data, form_char_data, lemma_char_data, morpheme_data = batch
        annotated_data = token_char_data.numpy(), form_char_data.numpy(), lemma_char_data.numpy(), morpheme_data.numpy()
        raw_data.append(dataset.get_raw_data(annotated_data, char_vocab, tag_vocab, feats_vocab))
    return raw_data


# def to_dataframe(raw_data_list, columns):
#     return pd.DataFrame(list(itertools.chain.from_iterable(raw_data_list)), columns=columns)
def to_dataset(data_samples):
    columns = ['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token', 'is_gold']
    return pd.DataFrame([row for sample in data_samples for row in sample], columns=columns)


data_path = Path('data')
raw_data_files = [data_path / f'{part}_{type(bert_tokenizer).__name__}_raw_data.csv' for part in partition]
if all([f.exists() for f in raw_data_files]):
    raw_train_path = data_path / f'train_{type(bert_tokenizer).__name__}_raw_data.csv'
    raw_dev_path = data_path / f'dev_{type(bert_tokenizer).__name__}_raw_data.csv'
    raw_test_path = data_path / f'test_{type(bert_tokenizer).__name__}_raw_data.csv'
    raw_train_data = pd.read_csv(raw_train_path)
    raw_dev_data = pd.read_csv(raw_dev_path)
    raw_test_data = pd.read_csv(raw_test_path)
else:
    raw_train_data = to_raw_data(train_dataloader, char_vocab, tag_vocab, feats_vocab)
    raw_dev_data = to_raw_data(dev_dataloader, char_vocab, tag_vocab, feats_vocab)
    raw_test_data = to_raw_data(test_dataloader, char_vocab, tag_vocab, feats_vocab)
    raw_train_path = data_path / f'train_{type(bert_tokenizer).__name__}_raw_data.csv'
    raw_dev_path = data_path / f'dev_{type(bert_tokenizer).__name__}_raw_data.csv'
    raw_test_path = data_path / f'test_{type(bert_tokenizer).__name__}_raw_data.csv'
    # columns = ['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token', 'is_gold']
    raw_train_data = to_dataset(raw_train_data)
    raw_dev_data = to_dataset(raw_dev_data)
    raw_test_data = to_dataset(raw_test_data)
    raw_train_data.to_csv(raw_train_path, index=False)
    raw_dev_data.to_csv(raw_dev_path, index=False)
    raw_test_data.to_csv(raw_test_path, index=False)


# Model
char_emb = nn.Embedding.from_pretrained(torch.tensor(char_vectors, dtype=torch.float),
                                        freeze=False, padding_idx=char_vocab['char2index']['<pad>'])
num_layers = 1
hidden_size = bert.config.hidden_size // num_layers
enc_dropout = 0.1
dec_dropout = 0.1
seg_out_size = len(char_vocab['char2index'])
out_dropout = 0.3
num_tags = len(tag_vocab['tag2index'])
token_tag_model = TokenSeq2SeqMorphTagger(char_emb=char_emb,
                                          hidden_size=hidden_size, num_layers=num_layers,
                                          enc_dropout=enc_dropout, dec_dropout=dec_dropout,
                                          out_size=seg_out_size, out_dropout=out_dropout, num_labels=num_tags)
morph_model = MorphTagModel(bert, bert_tokenizer, char_emb, token_tag_model)
device = None
if device is not None:
    morph_model.to(device)
print(morph_model)

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
parameters = list(filter(lambda p: p.requires_grad, morph_model.parameters()))
# parameters = seg_model.parameters()
adam = AdamW(parameters, lr=lr)
char_loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=char_vocab['char2index']['<pad>'])
tag_loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=tag_vocab['tag2index']['<pad>'])

lr_scheduler = get_linear_schedule_with_warmup(adam, num_warmup_steps=scheduler_warmup_steps,
                                               num_training_steps=num_training_steps)
# optimizer = ModelOptimizer(parameters, optimizer, optim_step_every, optim_max_grad_norm, lr_scheduler)


def _to_packed_form_char_data(form_chars, max_form_len, max_tag_len, eos, sep):
    form_chars = torch.split(form_chars, max_form_len, dim=1)
    token_mask = [torch.eq(c, eos) for c in form_chars]
    form_mask = [torch.eq(c, sep) for c in form_chars]
    max_tag_mask = [torch.lt(torch.cumsum(m, dim=1), max_tag_len) for m in form_mask]
    form_mask = [fm.long().add(tm.long()) > 1 for tm, fm in zip(max_tag_mask, form_mask)]
    form_mask = [fm.long().add(tm.long()) > 0 for tm, fm in zip(token_mask, form_mask)]
    token_mask = [torch.cumsum(tm, dim=1) - tm.long() for tm in token_mask]
    token_len = [torch.sum(torch.eq(tm, 0), dim=1) for tm in token_mask]
    token_form_mask = [torch.cumsum(fm[:, :tl], dim=1) - fm[:, :tl].long() for fm, tl in zip(form_mask, token_len)]
    token_form_len = [torch.unique(tfm, return_counts=True) for tfm in token_form_mask]
    form_len = torch.cat([tfl[1] for tfl in token_form_len])
    token_ids = [torch.tensor(tid + 1, dtype=torch.long).repeat(tl) for tid, tl in enumerate(token_len)]
    chars = [fc[:, :tl][0] for fc, tl in zip(form_chars, token_len)]
    morph_ids = [torch.tensor(mid + 1, dtype=torch.long).repeat(fl) for mid, fl in enumerate(form_len)]
    token_morph_ids, mid = [], 0
    for t in token_ids:
        m = morph_ids[mid]
        while len(t) > len(m):
            mid += 1
            next_m = morph_ids[mid]
            m = torch.cat([m, next_m])
        token_morph_ids.append(m)
        mid += 1
    data = [torch.stack([t, m, c ], dim=1) for t, m, c in zip(token_ids, token_morph_ids, chars)]
    return data


def _to_packed_lemma_char_data(form_char_data, char_vocab):
    morph_ids = [fcd[:, 1].unique() for fcd in form_char_data]
    lemma_len = [len(mid) for mid in morph_ids]
    token_ids = [torch.tensor(tid + 1, dtype=torch.long).repeat(tl) for tid, tl in enumerate(lemma_len)]
    chars = [torch.tensor([char_vocab['char2index']['_'] for l in range(ll)]) for ll in lemma_len]
    data = [torch.stack([t, m, c], dim=1) for t, m, c in zip(token_ids, morph_ids, chars)]
    return data


def _to_packed_morph_data(form_char_data, tags, max_tag_len, feats_vocab):
    tags = torch.split(tags, max_tag_len, dim=1)
    morph_ids = [fcd[:, 1].unique() for fcd in form_char_data]
    tag_len = [len(mid) for mid in morph_ids]
    token_ids = [torch.tensor(tid + 1, dtype=torch.long).repeat(tl) for tid, tl in enumerate(tag_len)]
    tag_ids = [t[:, :tl][0] for t, tl in zip(tags, tag_len)]
    tag_morph_ids, cur_id = [], 1
    for t in token_ids:
        tag_morph_ids.append(torch.tensor([cur_id + i for i, t1 in enumerate(t)]))
        cur_id += len(t)
    feats_value = feats_vocab['feats2index']['_']
    feats_ids = [torch.full_like(t, fill_value=feats_value) for t in tag_ids]
    data = [torch.stack([t, m, g, f], dim=1) for t, m, g, f in zip(token_ids, tag_morph_ids, tag_ids, feats_ids)]
    return data


def _to_char_data_sample(packed_char_data, sent_id):
    packed_token_ids = [fcd[:, 0].unique().item() for fcd in packed_char_data]
    packed_len = [len(fcd) for fcd in packed_char_data]
    padded_data = torch.nn.utils.rnn.pad_sequence(packed_char_data, batch_first=True, padding_value=0)
    for i in range(len(padded_data)):
        padded_data[i, packed_len[i]:, 0] = packed_token_ids[i]
        padded_data[i, packed_len[i]:, 1] = -1
        padded_data[i, packed_len[i]:, 2] = char_vocab['char2index']['<pad>']
    # sent_id, token_id, morph_id, char, total_len, token_len, max_num_tokens, max_form_len
    num_tokens = padded_data.shape[0]
    token_len = padded_data.shape[1]
    sent_id_data = sent_id.repeat(num_tokens, token_len)
    token_len_data = torch.tensor([packed_len]).T.repeat(1, token_len)
    total_len_data = torch.tensor(num_tokens * token_len).repeat(num_tokens, token_len)
    max_num_tokens_data = torch.tensor(num_tokens).repeat(num_tokens, token_len)
    max_len_data = torch.max(padded_data[:, :, 1]).repeat(num_tokens, token_len)
    data = torch.cat([sent_id_data.unsqueeze(2), padded_data, total_len_data.unsqueeze(2), token_len_data.unsqueeze(2),
                      max_num_tokens_data.unsqueeze(2), max_len_data.unsqueeze(2)], dim=2)
    return data.view(1, -1, data.shape[-1])


def _to_morph_data_sample(packed_morph_data, sent_id):
    packed_token_ids = [fcd[:, 0].unique().item() for fcd in packed_morph_data]
    packed_len = [len(fcd) for fcd in packed_morph_data]
    padded_data = torch.nn.utils.rnn.pad_sequence(packed_morph_data, batch_first=True, padding_value=0)
    for i in range(len(padded_data)):
        padded_data[i, packed_len[i]:, 0] = packed_token_ids[i]
        padded_data[i, packed_len[i]:, 1] = -1
        padded_data[i, packed_len[i]:, 2] = tag_vocab['tag2index']['<pad>']
        padded_data[i, packed_len[i]:, 3] = feats_vocab['feats2index']['<pad>']
    # sent_id, token_id, morph_id, tag, feats, total_len, token_len (num morphemes), max_num_tokens, max_morph_len
    num_tokens = padded_data.shape[0]
    token_len = padded_data.shape[1]
    sent_id_data = sent_id.repeat(num_tokens, token_len)
    token_len_data = torch.tensor([packed_len]).T.repeat(1, token_len)
    total_len_data = torch.tensor(num_tokens * token_len).repeat(num_tokens, token_len)
    max_num_tokens_data = torch.tensor(num_tokens).repeat(num_tokens, token_len)
    max_len_data = torch.tensor(token_len).repeat(num_tokens, token_len)
    data = torch.cat([sent_id_data.unsqueeze(2), padded_data, total_len_data.unsqueeze(2),  token_len_data.unsqueeze(2),
                      max_num_tokens_data.unsqueeze(2), max_len_data.unsqueeze(2)],  dim=2)
    return data.view(1, -1, data.shape[-1])


def to_data_sample(token_char_data, form_chars, max_form_len, tags, max_tags_len, eos, sep):
    packed_form_char_data =_to_packed_form_char_data(form_chars, max_form_len, max_tags_len, eos, sep)
    packed_lemma_char_data = _to_packed_lemma_char_data(packed_form_char_data, char_vocab)
    packed_morph_data = _to_packed_morph_data(packed_form_char_data, tags, max_tags_len, feats_vocab)
    sent_id = token_char_data[:, 0, 0]
    form_char_data = _to_char_data_sample(packed_form_char_data, sent_id)
    lemma_char_data = _to_char_data_sample(packed_lemma_char_data, sent_id)
    morph_data = _to_morph_data_sample(packed_morph_data, sent_id)
    return token_char_data.numpy(), form_char_data.numpy(), lemma_char_data.numpy(), morph_data.numpy()


def print_sample(sample_data):
    df = to_dataset([sample_data])
    sample = [str((s[1][0], s[1][1])) for s in df[['form', 'tag']].iterrows()]
    print(' '.join(sample))


# def to_dataset(data_samples):
#     columns = ['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token', 'is_gold']
#     return pd.DataFrame([row for sample in data_samples for row in sample], columns=columns)


def print_eval_scores(truth_df, decoded_df, step):
    if len(truth_df['sent_id'].unique()) != len(decoded_df['sent_id'].unique()):
        truth_df = truth_df.loc[truth_df['sent_id'].isin(decoded_df.sent_id.unique().tolist())]
    # aligned_scores = tb.seg_eval(truth_df, decoded_df, False)
    # mset_scores = tb.seg_eval(truth_df, decoded_df, True)
    aligned_scores = tb.morph_eval(truth_df, decoded_df, ['form'], False)
    mset_scores = tb.morph_eval(truth_df, decoded_df, ['form'], True)
    print(f'seg eval {step} aligned: [P: {aligned_scores[0]}, R: {aligned_scores[1]}, F: {aligned_scores[2]}]')
    print(f'seg eval {step} mset   : [P: {mset_scores[0]}, R: {mset_scores[1]}, F: {mset_scores[2]}]')

    aligned_scores = tb.morph_eval(truth_df, decoded_df, ['tag'], False)
    mset_scores = tb.morph_eval(truth_df, decoded_df, ['tag'], True)
    print(f'tag eval {step} aligned: [P: {aligned_scores[0]}, R: {aligned_scores[1]}, F: {aligned_scores[2]}]')
    print(f'tag eval {step} mset   : [P: {mset_scores[0]}, R: {mset_scores[1]}, F: {mset_scores[2]}]')

    aligned_scores = tb.morph_eval(truth_df, decoded_df, ['form', 'tag'], False)
    mset_scores = tb.morph_eval(truth_df, decoded_df, ['form', 'tag'], True)
    print(f'joint seg-tag eval {step} aligned: [P: {aligned_scores[0]}, R: {aligned_scores[1]}, F: {aligned_scores[2]}]')
    print(f'joint seg-tag eval {step} mset   : [P: {mset_scores[0]}, R: {mset_scores[1]}, F: {mset_scores[2]}]')


def process(epoch, phase, print_every, model, data, target_dataset, teacher_forcing_ratio, char_criterion,
            tag_criterion, optimizer=None, scheduler=None, max_grad_norm=None):
    print_form_char_loss, print_tag_loss, total_form_char_loss, total_tag_loss = 0, 0, 0, 0
    print_data_samples, total_data_samples = [], []
    sos = torch.tensor([char_vocab['char2index']['<s>']], dtype=torch.long, device=device)
    eos = torch.tensor([char_vocab['char2index']['</s>']], dtype=torch.long, device=device)
    sep = torch.tensor([char_vocab['char2index']['<sep>']], dtype=torch.long, device=device)
    pad = torch.tensor([char_vocab['char2index']['<pad>']], dtype=torch.long, device=device)
    special_symbols = {'<s>': sos, '</s>': eos, '<sep>': sep, '<pad>': pad}
    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        tokens_data, token_char_data, form_char_data, lemma_char_data, morpheme_data = batch
        input_xtokens = tokens_data[:, :, 1]
        input_token_chars = token_char_data[:, :, 1:]
        target_form_chars = form_char_data[:, :, 3:]
        target_tags = morpheme_data[:, :, 3:]
        max_form_len = target_form_chars[0, 0, -1]
        max_tags_len = target_tags[0, 0, -1]
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            form_char_scores, tag_scores = model(input_xtokens, input_token_chars, special_symbols, max_form_len,
                                                 max_tags_len, target_form_chars, target_tags)
        else:
            form_char_scores, tag_scores = model(input_xtokens, input_token_chars, special_symbols, max_form_len,
                                                 max_tags_len, None, None)
        target_form_chars_len = target_form_chars[0, 0, 1]
        target_tags_len = target_tags[0, 0, 2]
        target_chars = target_form_chars[:, :target_form_chars_len]
        target_tags = target_tags[:, :target_tags_len]
        form_char_loss = char_criterion(*(model.char_loss_prepare(form_char_scores, target_chars)))
        tag_loss = tag_criterion(*(model.tag_loss_prepare(tag_scores, target_tags)))
        print_form_char_loss += form_char_loss
        print_tag_loss += tag_loss
        if optimizer is not None:
            form_char_loss.backward(retain_graph=True)
            tag_loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        predicted_form_chars = model.decode(form_char_scores)
        predicted_tags = model.decode(tag_scores)
        # predicted_form_chars = target_form_chars[:, :target_form_chars_len, 0]
        # predicted_tags = target_tags[:, :target_tags_len, 0]
        predicted_data_sample = to_data_sample(token_char_data.cpu(), predicted_form_chars.cpu(), max_form_len.cpu(),
                                               predicted_tags.cpu(), max_tags_len.cpu(), eos.cpu(), sep.cpu())
        print_data_samples.append(dataset.get_raw_data(predicted_data_sample, char_vocab, tag_vocab, feats_vocab))
        if (i + 1) % print_every == 0:
            print(f'epoch {epoch} {phase}, step {i + 1} form char loss: {print_form_char_loss / print_every}')
            print(f'epoch {epoch} {phase}, step {i + 1} tag loss: {print_tag_loss / print_every}')
            print_sample(print_data_samples[-1])
            predicted_dataset = to_dataset(print_data_samples)
            print_eval_scores(target_dataset, predicted_dataset, i + 1)
            total_form_char_loss += print_form_char_loss
            total_tag_loss += print_tag_loss
            total_data_samples.extend(print_data_samples)
            print_form_char_loss, print_tag_loss = 0, 0
            print_data_samples = []
    if len(print_data_samples) > 0:
        total_form_char_loss += print_form_char_loss
        total_tag_loss += print_tag_loss
        total_data_samples.extend(print_data_samples)
    print(f'epoch {epoch} {phase}, total form char loss: {total_form_char_loss / len(data)}')
    print(f'epoch {epoch} {phase}, total tag loss: {total_tag_loss / len(data)}')
    predicted_dataset = to_dataset(total_data_samples)
    print_eval_scores(target_dataset, predicted_dataset, 'total')
    return predicted_dataset


teacher_forcing_ratio = 1.0
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    morph_model.train()
    process(epoch, 'train', 10, morph_model, train_dataloader, raw_train_data, teacher_forcing_ratio, char_loss_fct,
            tag_loss_fct, adam, lr_scheduler, max_grad_norm)
    morph_model.eval()
    with torch.no_grad():
        dev_samples = process(epoch, 'dev', 10, morph_model, dev_dataloader, raw_dev_data, 0.0, char_loss_fct,
                              tag_loss_fct)
        test_samples = process(epoch, 'test', 10, morph_model, test_dataloader, raw_test_data, 0.0, char_loss_fct,
                               tag_loss_fct)
