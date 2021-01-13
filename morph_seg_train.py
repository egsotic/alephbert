import random
import logging
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from transformers.models.bert import BertTokenizerFast, BertModel
from data import preprocess_morph_seg
from morph_seg_model import TokenCharSeq2SeqModel, MorphSegModel
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


# Data
raw_root_path = Path('data/raw/UD_Hebrew')
# raw_root_path = Path('data/raw/HebrewTreebank')
partition = tb.ud(raw_root_path, 'HTB')
# partition = tb.spmrl(raw_root_path, 'hebtb')

bert_version = 'bert-distilled-wordpiece-oscar-52000'
preprocessed_data_root_path = Path(f'data/preprocessed/UD_Hebrew/HTB/{bert_version}')
# preprocessed_root_path = Path(f'data/preprocessed/HebrewTreebank/hebtb/{bert_version}')


def load_preprocessed_data_samples(data_root_path, partition):
    logging.info('Loading preprocesssed form data samples')
    token_data = preprocess_morph_seg.load_token_data(data_root_path, partition)
    seg_data = preprocess_morph_seg.load_morph_seg_data(data_root_path, partition)
    datasets = {}
    for part in partition:
        token_arr, token_char_arr = token_data[part]
        morph_form_char_arr = seg_data[part]
        token_tensor = torch.tensor(token_arr, dtype=torch.long)
        token_char_tensor = torch.tensor(token_char_arr, dtype=torch.long)
        morph_form_char_tensor = torch.tensor(morph_form_char_arr, dtype=torch.long)
        datasets[part] = TensorDataset(token_tensor, token_char_tensor, morph_form_char_tensor)
    return datasets


datasets = {}
data_samples_file_paths = {part: preprocessed_data_root_path / f'{part}_form_data_samples.pt' for part in partition}
if all([data_samples_file_paths[part].exists() for part in data_samples_file_paths]):
    for part in partition:
        file_path = data_samples_file_paths[part]
        logging.info(f'Loading tensor dataset to file {file_path}')
        datasets[part] = torch.load(file_path)
else:
    datasets = load_preprocessed_data_samples(preprocessed_data_root_path, partition)
    for part in datasets:
        file_path = data_samples_file_paths[part]
        logging.info(f'Saving tensor dataset to file {file_path}')
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
char_vectors, char_vocab = preprocess_morph_seg.load_char_vocab(preprocessed_data_root_path)
char_emb = nn.Embedding.from_pretrained(torch.tensor(char_vectors, dtype=torch.float), freeze=False,
                                        padding_idx=char_vocab['char2index']['<pad>'])
num_layers = 2
hidden_size = bert.config.hidden_size // num_layers
dropout = 0.1
char_size = len(char_vocab['char2index'])
out_dropout = 0.5
token_char_s2s = TokenCharSeq2SeqModel(char_emb=char_emb, hidden_size=hidden_size, num_layers=num_layers,
                                       dropout=dropout, out_size=char_size, out_dropout=out_dropout)
morph_model = MorphSegModel(bert, bert_tokenizer, token_char_s2s)
device = None
if device is not None:
    morph_model.to(device)
print(morph_model)

# Special symbols
sos = torch.tensor([char_vocab['char2index']['<s>']], dtype=torch.long, device=device)
eos = torch.tensor([char_vocab['char2index']['</s>']], dtype=torch.long, device=device)
sep = torch.tensor([char_vocab['char2index']['<sep>']], dtype=torch.long, device=device)
pad = torch.tensor([char_vocab['char2index']['<pad>']], dtype=torch.long, device=device)
special_symbols = {'<s>': sos, '</s>': eos, '<sep>': sep, '<pad>': pad}


def to_sent_tokens(sent_token_chars):
    tokens = []
    for token_id in sent_token_chars[:, 1].unique():
        if token_id < 0:
            continue
        token_chars = sent_token_chars[sent_token_chars[:, 1] == token_id]
        token = ''.join([char_vocab['index2char'][c] for c in token_chars[:, 2].tolist()])
        tokens.append(token)
    return tokens


def to_sent_token_segments(sent_token_morph_chars):
    tokens = []
    token_mask = torch.nonzero(torch.eq(sent_token_morph_chars, eos))
    token_mask_map = {m[0].item(): m[1].item() for m in token_mask}
    for i, token_chars in enumerate(sent_token_morph_chars):
        token_len = token_mask_map[i] if i in token_mask_map else sent_token_morph_chars.shape[1]
        token_chars = token_chars[:token_len]
        form_mask = torch.nonzero(torch.eq(token_chars, sep))
        forms = []
        start_pos = 0
        for to_pos in form_mask:
            form_chars = token_chars[start_pos:to_pos]
            form = ''.join([char_vocab['index2char'][c.item()] for c in form_chars])
            forms.append(form)
            start_pos = to_pos + 1
        form_chars = token_chars[start_pos:]
        form = ''.join([char_vocab['index2char'][c.item()] for c in form_chars])
        forms.append(form)
        tokens.append(forms)
    return tokens


def morph_eval(decoded_sent_tokens, target_sent_tokens):
    aligned_decoded_counts, aligned_target_counts, aligned_intersection_counts = 0, 0, 0
    mset_decoded_counts, mset_target_counts, mset_intersection_counts = 0, 0, 0
    for decoded_tokens, target_tokens in zip(decoded_sent_tokens, target_sent_tokens):
        for decoded_segments, target_segments in zip(decoded_tokens, target_tokens):
            decoded_segment_counts, target_segment_counts = Counter(decoded_segments), Counter(target_segments)
            intersection_segment_counts = decoded_segment_counts & target_segment_counts
            mset_decoded_counts += sum(decoded_segment_counts.values())
            mset_target_counts += sum(target_segment_counts.values())
            mset_intersection_counts += sum(intersection_segment_counts.values())
            aligned_segments = [d for d, t in zip(decoded_segments, target_segments) if d == t]
            aligned_decoded_counts += len(decoded_segments)
            aligned_target_counts += len(target_segments)
            aligned_intersection_counts += len(aligned_segments)
    precision = aligned_intersection_counts / aligned_decoded_counts if aligned_decoded_counts else 0.0
    recall = aligned_intersection_counts / aligned_target_counts if aligned_target_counts else 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0.0
    aligned_scores = precision, recall, f1
    precision = mset_intersection_counts / mset_decoded_counts if mset_decoded_counts else 0.0
    recall = mset_intersection_counts / mset_target_counts if mset_target_counts else 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0.0
    mset_scores = precision, recall, f1
    return aligned_scores, mset_scores


def to_sent_token_seg_lattice_rows(sent_id, tokens, token_segments):
    rows = []
    node_id = 0
    for token_id, (token, forms) in enumerate(zip(tokens, token_segments)):
        for form in forms:
            row = [sent_id, node_id, node_id+1, form, '_', '_', '_', token_id+1, token, True]
            rows.append(row)
            node_id += 1
    return rows


def get_morph_seg_lattice_data(sent_token_seg_rows):
    lattice_rows = []
    for row in sent_token_seg_rows:
        lattice_rows.extend(to_sent_token_seg_lattice_rows(*row))
    return pd.DataFrame(lattice_rows,
                        columns=['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id',
                                 'token', 'is_gold'])


def print_eval_scores(decoded_df, truth_df, step):
    aligned_scores, mset_scores = tb.morph_eval(pred_df=decoded_df, gold_df=truth_df, fields=['form'])
    for fs in aligned_scores:
        p, r, f = aligned_scores[fs]
        print(f'eval {step} aligned {fs}: [P: {p}, R: {r}, F: {f}]')
        p, r, f = mset_scores[fs]
        print(f'eval {step} mset    {fs}: [P: {p}, R: {r}, F: {f}]')


# Training and evaluation routine
def process(epoch, phase, print_every, model, data, teacher_forcing_ratio, char_criterion, optimizer=None,
            max_grad_norm=None):
    print_form_char_loss, total_form_char_loss = 0, 0
    print_decoded_sent_tokens, total_decoded_sent_tokens = [], []
    print_target_sent_tokens, total_target_sent_tokens = [], []
    print_decoded_token_lattice_rows, total_decoded_token_lattice_rows = [], []

    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        tokens, token_chars, form_chars = batch
        input_xtokens = tokens[:, :, 1:]
        input_token_chars = token_chars[:, :, 1:]
        target_token_form_chars = form_chars[:, :, :, 1:]
        max_form_len = target_token_form_chars.shape[2]
        target_token_chars = target_token_form_chars[0, :, :, 1]
        target_chars = target_token_chars[target_token_chars[:, 0] != 0]

        sent_id = input_token_chars[:, :, 0].unique().item()
        sent_tokens = to_sent_tokens(token_chars[0])
        target_segments = to_sent_token_segments(target_chars.view(-1, max_form_len))
        print_target_sent_tokens.append(target_segments)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        num_tokens, form_char_scores = model(input_xtokens, input_token_chars, special_symbols, max_form_len,
                                             target_token_form_chars if use_teacher_forcing else None)
        target_chars = target_token_form_chars[:, :num_tokens, :, 1].view(-1)
        form_char_loss = char_criterion(form_char_scores.squeeze(0), target_chars.squeeze(0))
        print_form_char_loss += form_char_loss

        decoded_chars = model.decode(form_char_scores).squeeze(0)
        decoded_segments = to_sent_token_segments(decoded_chars.view(-1, max_form_len))
        print_decoded_sent_tokens.append(decoded_segments)
        decoded_token_lattice_rows = (sent_id, sent_tokens, decoded_segments)
        print_decoded_token_lattice_rows.append(decoded_token_lattice_rows)

        if optimizer is not None:
            form_char_loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % print_every == 0:
            print(f'epoch {epoch} {phase}, step {i + 1} form char loss: {print_form_char_loss / print_every}')
            print(decoded_segments)
            total_form_char_loss += print_form_char_loss
            print_form_char_loss = 0

            aligned_scores, mset_scores = morph_eval(print_decoded_sent_tokens, print_target_sent_tokens)
            print(aligned_scores)
            print(mset_scores)
            total_decoded_sent_tokens.extend(print_decoded_sent_tokens)
            total_target_sent_tokens.extend(print_target_sent_tokens)
            total_decoded_token_lattice_rows.extend(print_decoded_token_lattice_rows)
            print_decoded_sent_tokens = []
            print_target_sent_tokens = []
            print_decoded_token_lattice_rows = []
    if print_form_char_loss > 0:
        total_form_char_loss += print_form_char_loss
        total_target_sent_tokens.extend(print_target_sent_tokens)
        total_decoded_sent_tokens.extend(print_decoded_sent_tokens)
        total_decoded_token_lattice_rows.extend(print_decoded_token_lattice_rows)
    print(f'epoch {epoch} {phase}, total form char loss: {total_form_char_loss / len(data)}')
    aligned_scores, mset_scores = morph_eval(total_decoded_sent_tokens, total_target_sent_tokens)
    print(aligned_scores)
    print(mset_scores)
    return get_morph_seg_lattice_data(total_decoded_token_lattice_rows)


# Optimization
epochs = 1
max_grad_norm = 1.0
lr = 1e-3
# freeze bert
for param in bert.parameters():
    param.requires_grad = False
parameters = list(filter(lambda p: p.requires_grad, morph_model.parameters()))
# parameters = morph_model.parameters()
adam = optim.AdamW(parameters, lr=lr)
# char_loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=char_vocab['char2index']['<pad>'])
char_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
teacher_forcing_ratio = 1.0


# Training epochs
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    morph_model.train()
    process(epoch, 'train', 10, morph_model, train_dataloader, teacher_forcing_ratio, char_loss_fct, adam, max_grad_norm)
    morph_model.eval()
    with torch.no_grad():
        dev_samples = process(epoch, 'dev', 10, morph_model, dev_dataloader, 0.0, char_loss_fct)
        print_eval_scores(decoded_df=dev_samples, truth_df=partition['dev'], step=epoch)
        test_samples = process(epoch, 'test', 10, morph_model, test_dataloader, 0.0, char_loss_fct)
        print_eval_scores(decoded_df=test_samples, truth_df=partition['test'], step=epoch)
# dev_samples.to_csv(out_path / 'dev.lattices.csv', index=False)
# test_samples.to_csv(out_path / 'test.lattices.csv', index=False)
# torch.save(morph_model.state_dict(), out_path / 'model-state.pt')
# torch.save(morph_model, out_path / 'model.pt')


def test():
    # Output location
    out_path = Path(f'experiments/morph-seg/bert/distilled/wordpiece/{bert_version}/UD_Hebrew/HTB')
    # out_path = Path(f'experiments/morph-seg/bert/distilled/wordpiece/{bert_version}/HebrewTreebank/hebtb')
    m = torch.load(out_path / 'model.pt', map_location=torch.device('cpu'))
    m.eval()
    with torch.no_grad():
        dev_samples = process(0, 'dev', 100, m, dev_dataloader, 0.0, char_loss_fct)
        print_eval_scores(decoded_df=dev_samples, truth_df=partition['dev'], step=0)
        test_samples = process(m, 'test', 100, m, test_dataloader, 0.0, char_loss_fct)
        print_eval_scores(decoded_df=test_samples, truth_df=partition['test'], step=0)
