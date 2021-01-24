import random
import logging
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from transformers import BertModel
from data import preprocess_morph_seg
from seg_model import TokenCharSegmentDecoder, MorphSegModel
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

schema = "UD"
# schema = "SPMRL"
data_src = "UD_Hebrew"
# data_src = "HebrewTreebank"
# data_src = "for_amit_spmrl"
tb_name = "HTB"
# tb_name = "hebtb"
raw_root_path = Path(f'data/raw/{data_src}')

# Data
if tb_name == 'HTB':
    partition = tb.ud(raw_root_path, tb_name)
elif tb_name == 'hebtb':
    partition = tb.spmrl(raw_root_path, tb_name)
else:
    partition = {'train': None, 'dev': None, 'test': None}
bert_version = 'bert-distilled-wordpiece-oscar-52000'
preprocessed_data_root_path = Path(f'data/preprocessed/{data_src}/{tb_name}/{bert_version}')


def load_preprocessed_data_samples(data_root_path, partition):
    logging.info(f'Loading preprocesssed {schema} form data samples')
    xtoken_data = preprocess_morph_seg.load_xtoken_data(data_root_path, partition)
    token_char_data = preprocess_morph_seg.load_token_char_data(data_root_path, partition)
    form_char_data = preprocess_morph_seg.load_morph_seg_data(data_root_path, partition)
    datasets = {}
    for part in partition:
        xtoken_tensor = torch.tensor(xtoken_data[part][:, :, 1:], dtype=torch.long)
        token_char_tensor = torch.tensor(token_char_data[part][:, :, :, 1:], dtype=torch.long)
        morph_form_char_tensor = torch.tensor(form_char_data[part], dtype=torch.long)
        datasets[part] = TensorDataset(xtoken_tensor, token_char_tensor, morph_form_char_tensor)
    return datasets


datasets = {}
data_samples_file_paths = {part: preprocessed_data_root_path / f'{part}_form_data_samples.pt' for part in partition}
if all([data_samples_file_paths[part].exists() for part in data_samples_file_paths]):
    for part in partition:
        file_path = data_samples_file_paths[part]
        logging.info(f'Loading {schema} form tensor dataset to file {file_path}')
        datasets[part] = torch.load(file_path)
else:
    datasets = load_preprocessed_data_samples(preprocessed_data_root_path, partition)
    for part in datasets:
        file_path = data_samples_file_paths[part]
        logging.info(f'Saving {schema} form tensor dataset to file {file_path}')
        torch.save(datasets[part], file_path)
train_dataloader = DataLoader(datasets['train'], batch_size=10, shuffle=False)
dev_dataloader = DataLoader(datasets['dev'], batch_size=1)
test_dataloader = DataLoader(datasets['test'], batch_size=1)

# Language Model
bert_folder_path = Path(f'./experiments/transformers/bert/distilled/wordpiece/{bert_version}')
logging.info(f'BERT folder path: {str(bert_folder_path)}')
bert = BertModel.from_pretrained(str(bert_folder_path))
# bert_tokenizer = BertTokenizerFast.from_pretrained(str(bert_folder_path))
logging.info('BERT model and tokenizer loaded')

# Morph Segmentation Model
char_vectors, char_vocab = preprocess_morph_seg.load_char_vocab(preprocessed_data_root_path)
char_emb = nn.Embedding.from_pretrained(torch.tensor(char_vectors, dtype=torch.float), freeze=False,
                                        padding_idx=char_vocab['char2index']['<pad>'])
num_layers = 2
hidden_size = bert.config.hidden_size // num_layers
dropout = 0.1
num_chars = len(char_vocab['char2index'])
out_dropout = 0.5
seg_dec = TokenCharSegmentDecoder(char_emb=char_emb, hidden_size=hidden_size, num_layers=num_layers,
                                  dropout=dropout, out_size=num_chars, out_dropout=out_dropout)
morph_model = MorphSegModel(bert, seg_dec)
device = None
if device is not None:
    morph_model.to(device)
print(morph_model)

# Special symbols
sos = torch.tensor([char_vocab['char2index']['<s>']], dtype=torch.long)
eos = torch.tensor([char_vocab['char2index']['</s>']], dtype=torch.long)
sep = torch.tensor([char_vocab['char2index']['<sep>']], dtype=torch.long)
pad = torch.tensor([char_vocab['char2index']['<pad>']], dtype=torch.long)
special_symbols = {'<s>': sos.to(device), '</s>': eos.to(device), '<sep>': sep.to(device), '<pad>': pad.to(device)}


def to_sent_tokens(token_chars):
    tokens = []
    for chars in token_chars:
        token = ''.join([char_vocab['index2char'][c] for c in chars[chars > 0].tolist()])
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
            form_chars = token_chars[start_pos:to_pos.item()]
            form = ''.join([char_vocab['index2char'][c.item()] for c in form_chars])
            forms.append(form)
            start_pos = to_pos.item() + 1
        form_chars = token_chars[start_pos:]
        form = ''.join([char_vocab['index2char'][c.item()] for c in form_chars])
        forms.append(form)
        tokens.append(forms)
    return tokens


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


def print_eval_scores(decoded_df, truth_df, step):
    aligned_scores, mset_scores = tb.morph_eval(pred_df=decoded_df, gold_df=truth_df, fields=['form'])
    for fs in aligned_scores:
        p, r, f = aligned_scores[fs]
        print(f'eval {step} aligned {fs}: [P: {p}, R: {r}, F: {f}]')
        p, r, f = mset_scores[fs]
        print(f'eval {step} mset    {fs}: [P: {p}, R: {r}, F: {f}]')


# Training and evaluation routine
def process(model: MorphSegModel, data: DataLoader, criterion: nn.CrossEntropyLoss, epoch, phase, print_every,
            teacher_forcing_ratio=0.0, optimizer=None, max_grad_norm=None):
    print_loss, total_loss = 0, 0
    print_target_forms, total_target_forms = [], []
    print_decoded_forms, total_decoded_forms = [], []
    print_decoded_lattice_rows, total_decoded_lattice_rows = [], []

    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        batch_scores, batch_targets, batch_token_chars, batch_sent_ids, batch_num_tokens = [], [], [], [], []
        for sent_token_ctx, sent_token_chars, sent_form_chars in zip(model.embed(batch[0]), batch[1], batch[2]):
            input_token_chars = sent_token_chars[:, :, -1]
            num_tokens = len(sent_token_chars[sent_token_chars[:, 0, 1] > 0])
            target_token_form_chars = sent_form_chars[:, :, -1]
            max_form_len = target_token_form_chars.shape[1]
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            sent_scores, _ = model(sent_token_ctx, input_token_chars, special_symbols, num_tokens, max_form_len,
                                   target_token_form_chars if use_teacher_forcing else None)
            batch_scores.append(sent_scores)
            batch_targets.append(target_token_form_chars[:num_tokens])
            batch_token_chars.append(input_token_chars[:num_tokens])
            batch_sent_ids.append(sent_form_chars[:, :, 0].unique().item())
            batch_num_tokens.append(num_tokens)
        batch_targets = nn.utils.rnn.pad_sequence(batch_targets, batch_first=True)
        batch_scores = nn.utils.rnn.pad_sequence(batch_scores, batch_first=True)
        loss_batch_targets = batch_targets.view(-1)
        loss_batch_scores = batch_scores.view(-1, batch_scores.shape[-1])
        loss = criterion(loss_batch_scores, loss_batch_targets)
        print_loss += loss.item()
        if optimizer is not None:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            batch_decoded_chars = model.decode(batch_scores)
        for sent_id, input_chars, target_chars, decoded_chars, num_tokens in zip(batch_sent_ids, batch_token_chars,
                                                                               batch_targets, batch_decoded_chars,
                                                                               batch_num_tokens):
            input_chars = input_chars.to('cpu')
            target_chars = target_chars[:num_tokens].to('cpu')
            decoded_chars = decoded_chars[:num_tokens].to('cpu')
            input_tokens = to_sent_tokens(input_chars)
            target_segments = to_sent_token_segments(target_chars)
            decoded_segments = to_sent_token_segments(decoded_chars)
            decoded_token_lattice_rows = (sent_id, input_tokens, decoded_segments)
            print_target_forms.append(target_segments)
            print_decoded_forms.append(decoded_segments)
            print_decoded_lattice_rows.append(decoded_token_lattice_rows)

        if (i + 1) % print_every == 0:
            sent_id, input_tokens, decoded_segments = print_decoded_lattice_rows[-1]
            target_segments = print_target_forms[-1]
            print(f'epoch {epoch} {phase}, step {i + 1} form char loss: {print_loss / print_every}')
            print(f'sent #{sent_id} input tokens  : {input_tokens}')
            print(f'sent #{sent_id} target forms  : {list(reversed(target_segments))}')
            print(f'sent #{sent_id} decoded forms : {list(reversed(decoded_segments))}')
            total_loss += print_loss
            print_loss = 0

            aligned_scores, mset_scores = morph_eval(print_decoded_forms, print_target_forms)
            print(aligned_scores)
            print(mset_scores)
            total_decoded_forms.extend(print_decoded_forms)
            total_target_forms.extend(print_target_forms)
            total_decoded_lattice_rows.extend(print_decoded_lattice_rows)
            print_target_forms = []
            print_decoded_forms = []
            print_decoded_lattice_rows = []
    if print_loss > 0:
        total_loss += print_loss
        total_target_forms.extend(print_target_forms)
        total_decoded_forms.extend(print_decoded_forms)
        total_decoded_lattice_rows.extend(print_decoded_lattice_rows)
    print(f'epoch {epoch} {phase}, total form char loss: {total_loss / len(data)}')
    aligned_scores, mset_scores = morph_eval(total_decoded_forms, total_target_forms)
    print(aligned_scores)
    print(mset_scores)
    return get_morph_seg_lattice_data(total_decoded_lattice_rows)


# Optimization
epochs = 3
max_grad_norm = 1.0
lr = 1e-2
# freeze bert
for param in bert.parameters():
    param.requires_grad = False
parameters = list(filter(lambda p: p.requires_grad, morph_model.parameters()))
# parameters = morph_model.parameters()
adam = optim.AdamW(parameters, lr=lr)
loss_fct = nn.CrossEntropyLoss(ignore_index=0)
teacher_forcing_ratio = 1.0


# Training epochs
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    morph_model.train()
    process(morph_model, train_dataloader, loss_fct, epoch, 'train', 10, teacher_forcing_ratio, adam, max_grad_norm)
    morph_model.eval()
    with torch.no_grad():
        dev_samples = process(morph_model, dev_dataloader, loss_fct, epoch, 'dev', 100)
        print_eval_scores(decoded_df=dev_samples, truth_df=partition['dev'], step=epoch)
        test_samples = process(morph_model, test_dataloader, loss_fct, epoch, 'test', 100)
        print_eval_scores(decoded_df=test_samples, truth_df=partition['test'], step=epoch)
# dev_samples.to_csv(out_path / 'dev.lattices.csv', index=False)
# test_samples.to_csv(out_path / 'test.lattices.csv', index=False)
# torch.save(morph_model.state_dict(), out_path / 'morph-model-state.pt')
# torch.save(morph_model, out_path / 'morph-model.pt')


def test():
    out_path = Path(f'experiments/morph-seg/bert/distilled/wordpiece/{bert_version}/UD_Hebrew/HTB')
    # out_path = Path(f'experiments/morph-seg/bert/distilled/wordpiece/{bert_version}/HebrewTreebank/hebtb')
    m = torch.load(out_path / 'morph-model.pt', map_location=torch.device('cpu'))
    m.eval()
    with torch.no_grad():
        dev_samples = process(m, dev_dataloader, loss_fct, epoch, 'dev', 10)
        print_eval_scores(decoded_df=dev_samples, truth_df=partition['dev'], step=epoch)
        test_samples = process(m, test_dataloader, loss_fct, epoch, 'test', 10)
        print_eval_scores(decoded_df=test_samples, truth_df=partition['test'], step=epoch)