import random
import logging
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from transformers.models.bert import BertTokenizerFast, BertModel
from data import preprocess_morph_tag
from token_tag_model import TaggerModel, TokenTagsDecoder
from collections import Counter
import pandas as pd
from bclm import treebank as tb

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

raw_root_path = Path('data/raw/for_amit_spmrl')
partition = tb.ud(raw_root_path, 'hebtb')
bert_version = 'bert-distilled-wordpiece-oscar-52000'
preprocessed_data_root_path = Path(f'data/preprocessed/for_amit_spmrl/hebtb/{bert_version}')


def load_preprocessed_data_samples(data_root_path, partition):
    logging.info('Loading preprocessed SPMRL NER data samples')
    token_data = preprocess_morph_tag.load_token_data(data_root_path, partition)
    tag_data = preprocess_morph_tag.load_morph_tag_data(data_root_path, partition)
    datasets = {}
    for part in partition:
        xtoken_arr, token_char_arr = token_data[part]
        tag_arr = tag_data[part]
        xtoken_tensor = torch.tensor(xtoken_arr, dtype=torch.long)
        token_char_tensor = torch.tensor(token_char_arr, dtype=torch.long)
        tag_tensor = torch.tensor(tag_arr, dtype=torch.long)
        datasets[part] = TensorDataset(xtoken_tensor, token_char_tensor, tag_tensor)
    return datasets


datasets = {}
data_samples_file_paths = {part: preprocessed_data_root_path / f'{part}_spmrl_ner_data_samples.pt' for part in partition}
if all([data_samples_file_paths[part].exists() for part in data_samples_file_paths]):
    for part in partition:
        file_path = data_samples_file_paths[part]
        logging.info(f'Loading SPMRL NER tensor dataset to file {file_path}')
        datasets[part] = torch.load(file_path)
else:
    datasets = load_preprocessed_data_samples(preprocessed_data_root_path, partition)
    for part in datasets:
        file_path = data_samples_file_paths[part]
        logging.info(f'Saving SPMRL NER tensor dataset to file {file_path}')
        torch.save(datasets[part], file_path)

train_dataloader = DataLoader(datasets['train'], batch_size=1, shuffle=False)
dev_dataloader = DataLoader(datasets['dev'], batch_size=1)
test_dataloader = DataLoader(datasets['test'], batch_size=1)

# Language Model
bert_folder_path = Path(f'./experiments/transformers/bert/distilled/wordpiece/{bert_version}')
logging.info(f'BERT folder path: {str(bert_folder_path)}')
bert = BertModel.from_pretrained(str(bert_folder_path))
bert_tokenizer = BertTokenizerFast.from_pretrained(str(bert_folder_path))
logging.info('BERT model and tokenizer loaded')

# Morph Segmentation Model
with_tag_seq_symbols = True
char_vectors, char_vocab, tag_vocab, feats_vocab = preprocess_morph_tag.load_morph_vocab(preprocessed_data_root_path,
                                                                                         list(partition.keys()),
                                                                                         with_tag_seq_symbols)
tag_size = len(tag_vocab['tag2index'])
tag_emb = nn.Embedding(num_embeddings=tag_size, embedding_dim=50, padding_idx=tag_vocab['tag2index']['<pad>'])
num_layers = 2
hidden_size = bert.config.hidden_size // num_layers
dropout = 0.1
out_dropout = 0.5
tag_decoder = TokenTagsDecoder(tag_emb, hidden_size, num_layers, dropout, tag_size, out_dropout)
tagger_model = TaggerModel(bert, bert_tokenizer, tag_decoder)
device = None
if device is not None:
    tagger_model.to(device)
print(tagger_model)

# Special symbols
sos = torch.tensor([tag_vocab['tag2index']['<s>']], dtype=torch.long, device=device)
eos = torch.tensor([tag_vocab['tag2index']['</s>']], dtype=torch.long, device=device)
pad = torch.tensor([tag_vocab['tag2index']['<pad>']], dtype=torch.long, device=device)
special_symbols = {'<s>': sos, '</s>': eos, '<pad>': pad}


def to_sent_token_tags(sent_token_tags):
    tokens = []
    token_mask = torch.nonzero(torch.eq(sent_token_tags, eos))
    token_mask_map = {m[0].item(): m[1].item() for m in token_mask}
    for i, token_tags in enumerate(sent_token_tags):
        token_len = token_mask_map[i] if i in token_mask_map else sent_token_tags.shape[1]
        token_tags = token_tags[:token_len]
        tags = [tag_vocab['index2tag'][t.item()] for t in token_tags]
        tokens.append(tags)
    return tokens


def morph_eval(decoded_sent_tokens, target_sent_tokens):
    aligned_decoded_counts, aligned_target_counts, aligned_intersection_counts = 0, 0, 0
    mset_decoded_counts, mset_target_counts, mset_intersection_counts = 0, 0, 0
    for decoded_tokens, target_tokens in zip(decoded_sent_tokens, target_sent_tokens):
        for decoded_tags, target_tags in zip(decoded_tokens, target_tokens):
            decoded_tag_counts, target_tag_counts = Counter(decoded_tags), Counter(target_tags)
            intersection_tag_counts = decoded_tag_counts & target_tag_counts
            mset_decoded_counts += sum(decoded_tag_counts.values())
            mset_target_counts += sum(target_tag_counts.values())
            mset_intersection_counts += sum(intersection_tag_counts.values())
            aligned_tags = [d for d, t in zip(decoded_tags, target_tags) if d == t]
            aligned_decoded_counts += len(decoded_tags)
            aligned_target_counts += len(target_tags)
            aligned_intersection_counts += len(aligned_tags)
    precision = aligned_intersection_counts / aligned_decoded_counts if aligned_decoded_counts else 0.0
    recall = aligned_intersection_counts / aligned_target_counts if aligned_target_counts else 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0.0
    aligned_scores = precision, recall, f1
    precision = mset_intersection_counts / mset_decoded_counts if mset_decoded_counts else 0.0
    recall = mset_intersection_counts / mset_target_counts if mset_target_counts else 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0.0
    mset_scores = precision, recall, f1
    return aligned_scores, mset_scores


def to_sent_tokens(sent_token_chars):
    tokens = []
    for token_id in sent_token_chars[:, 1].unique():
        if token_id < 0:
            continue
        token_chars = sent_token_chars[sent_token_chars[:, 1] == token_id]
        token = ''.join([char_vocab['index2char'][c] for c in token_chars[:, 2].tolist()])
        tokens.append(token)
    return tokens


def to_sent_token_seg_lattice_rows(sent_id, tokens, token_tags):
    rows = []
    node_id = 0
    for token_id, (token, tags) in enumerate(zip(tokens, token_tags)):
        for tag in tags:
            row = [sent_id, node_id, node_id+1, '_', '_', tag, '_', token_id+1, token, True]
            rows.append(row)
            node_id += 1
    return rows


def get_morph_tag_lattice_data(sent_token_seg_rows):
    lattice_rows = []
    for row in sent_token_seg_rows:
        lattice_rows.extend(to_sent_token_seg_lattice_rows(*row))
    return pd.DataFrame(lattice_rows,
                        columns=['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id',
                                 'token', 'is_gold'])


def print_eval_scores(decoded_df, truth_df, step):
    aligned_scores, mset_scores = tb.morph_eval(pred_df=decoded_df, gold_df=truth_df, fields=['tag'])
    for fs in aligned_scores:
        p, r, f = aligned_scores[fs]
        print(f'eval {step} aligned {fs}: [P: {p}, R: {r}, F: {f}]')
        p, r, f = mset_scores[fs]
        print(f'eval {step} mset    {fs}: [P: {p}, R: {r}, F: {f}]')


# Training and evaluation routine
def process(model: TaggerModel, data: DataLoader, criterion: nn.CrossEntropyLoss, epoch, phase, print_every,
            teacher_forcing_ratio=0.0, optimizer=None, max_grad_norm=None):
    print_tag_loss, total_tag_loss = 0, 0
    print_decoded_sent_tokens, total_decoded_sent_tokens = [], []
    print_target_sent_tokens, total_target_sent_tokens = [], []
    print_decoded_token_lattice_rows, total_decoded_token_lattice_rows = [], []

    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        xtokens, token_chars, tags = batch
        input_xtokens = xtokens[:, :, 1:]
        target_token_tags = tags[:, :, :, 1:]
        max_tag_len = target_token_tags.shape[2]

        sent_id = tags[:, :, :, 0].unique().item()
        sent_tokens = to_sent_tokens(token_chars[0])
        target_tags = target_token_tags[target_token_tags[:, :, :, 0] > 0]
        target_tags = to_sent_token_tags(target_tags[:, 1].view(-1, max_tag_len))
        print_target_sent_tokens.append(target_tags)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        num_tokens, tag_scores = model(input_xtokens, special_symbols, max_tag_len, target_token_tags if use_teacher_forcing else None)
        target_tags = target_token_tags[:, :num_tokens, :, 1].view(-1)
        tag_loss = criterion(tag_scores.squeeze(0), target_tags.squeeze(0))
        print_tag_loss += tag_loss

        decoded_tags = model.decode(tag_scores).squeeze(0)
        decoded_tags = to_sent_token_tags(decoded_tags.view(-1, max_tag_len))
        print_decoded_sent_tokens.append(decoded_tags)
        decoded_token_lattice_rows = (sent_id, sent_tokens, decoded_tags)
        print_decoded_token_lattice_rows.append(decoded_token_lattice_rows)

        if optimizer is not None:
            tag_loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % print_every == 0:
            print(f'epoch {epoch} {phase}, step {i + 1} tag loss: {print_tag_loss / print_every}')
            print(f'sent #{sent_id} tokens: {sent_tokens}')
            print(f'sent #{sent_id} decoded_tags: {decoded_tags}')
            total_tag_loss += print_tag_loss
            print_tag_loss = 0

            aligned_scores, mset_scores = morph_eval(print_decoded_sent_tokens, print_target_sent_tokens)
            print(aligned_scores)
            print(mset_scores)
            total_decoded_sent_tokens.extend(print_decoded_sent_tokens)
            total_target_sent_tokens.extend(print_target_sent_tokens)
            total_decoded_token_lattice_rows.extend(print_decoded_token_lattice_rows)
            print_decoded_sent_tokens = []
            print_target_sent_tokens = []
            print_decoded_token_lattice_rows = []
    if print_tag_loss > 0:
        total_tag_loss += print_tag_loss
        total_target_sent_tokens.extend(print_target_sent_tokens)
        total_decoded_sent_tokens.extend(print_decoded_sent_tokens)
        total_decoded_token_lattice_rows.extend(print_decoded_token_lattice_rows)
    print(f'epoch {epoch} {phase}, total tag loss: {total_tag_loss / len(data)}')
    aligned_scores, mset_scores = morph_eval(total_decoded_sent_tokens, total_target_sent_tokens)
    print(aligned_scores)
    print(mset_scores)
    return get_morph_tag_lattice_data(total_decoded_token_lattice_rows)


# Optimization
epochs = 1
max_grad_norm = 1.0
lr = 1e-3
# freeze bert
for param in bert.parameters():
    param.requires_grad = False
parameters = list(filter(lambda p: p.requires_grad, tagger_model.parameters()))
# parameters = morph_model.parameters()
adam = optim.AdamW(parameters, lr=lr)
# char_loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=char_vocab['char2index']['<pad>'])
loss_fct = nn.CrossEntropyLoss(ignore_index=0)
teacher_forcing_ratio = 1.0


# Training epochs
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    tagger_model.train()
    process(tagger_model, train_dataloader, loss_fct, epoch, 'train', 10, teacher_forcing_ratio, adam,  max_grad_norm)
    tagger_model.eval()
    with torch.no_grad():
        dev_samples = process(tagger_model, dev_dataloader, loss_fct, epoch, 'dev', 10)
        print_eval_scores(decoded_df=dev_samples, truth_df=partition['dev'], step=epoch)
        test_samples = process(tagger_model, test_dataloader, loss_fct, epoch, 'test', 10)
        print_eval_scores(decoded_df=test_samples, truth_df=partition['test'], step=epoch)
