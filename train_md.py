import itertools
import random
import logging
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from transformers import BertModel, BertTokenizerFast
from data import preprocess_form, preprocess_labels2
from model_md import BertTokenEmbeddingModel, SegmentDecoder, MorphSequenceModel, MorphPipelineModel
from collections import Counter
import pandas as pd
from bclm import treebank as tb, ne_evaluate_mentions
from hebrew_root_tokenizer import AlefBERTRootTokenizer

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

# Config
tb_schema = "UD"
# schema = "SPMRL"
# tb_data_src = "UD_Hebrew"
tb_data_src = "for_amit_spmrl"
# tb_data_src = "HebrewTreebank"
# tb_name = "HTB"
tb_name = "hebtb"

bert_tokenizer_type = 'wordpiece'
# bert_tokenizer_type = 'roots'
bert_vocab_size = 52000
# bert_vocab_size = 10000
bert_corpus_name = 'oscar'
bert_model_type = 'distilled'
# bert_version = 'hebert'
# bert_version = 'mbert'
# bert_version = 'mbert-cased'
bert_version = f'bert-{bert_model_type}-{bert_tokenizer_type}-{bert_corpus_name}-{bert_vocab_size}'

# md_strategry = "morph-pipeline"
md_strategry = "morph-sequence"
# md_strategry = "morph-segment-only"

# Data
raw_root_path = Path(f'data/raw/{tb_data_src}')
if tb_name == 'HTB':
    partition = tb.ud(raw_root_path, tb_name)
elif tb_name == 'hebtb':
    if tb_schema == "UD":
        partition = tb.spmrl_conllu(raw_root_path, tb_name)
    else:
        partition = tb.spmrl(raw_root_path, tb_name)
else:
    partition = {'train': None, 'dev': None, 'test': None}
preprocessed_data_root_path = Path(f'data/preprocessed/{tb_data_src}/{tb_name}/{bert_version}')


def load_preprocessed_data_samples(data_root_path, partition, label_names) -> dict:
    logging.info(f'Loading preprocesssed {tb_schema} form tag data samples')
    xtoken_data = preprocess_form.load_xtoken_data(data_root_path, partition)
    token_char_data = preprocess_form.load_token_char_data(data_root_path, partition)
    form_char_data = preprocess_form.load_form_data(data_root_path, partition)
    label_data = preprocess_labels2.load_labeled_data(data_root_path, partition, label_names=label_names)
    datasets = {}
    for part in partition:
        xtoken_tensor = torch.tensor(xtoken_data[part][:, :, 1:], dtype=torch.long)
        token_char_tensor = torch.tensor(token_char_data[part][:, :, :, 1:], dtype=torch.long)
        form_char_tensor = torch.tensor(form_char_data[part], dtype=torch.long)
        label_tensor = torch.tensor(label_data[part], dtype=torch.long)
        datasets[part] = TensorDataset(xtoken_tensor, token_char_tensor, form_char_tensor, label_tensor)
    return datasets


datasets = {}
# label_names = ['tag']
label_names = ['tag', 'ner']
# label_names = ['tag', 'Gender', 'Number', 'Person', 'Tense']
# label_names = None
if label_names is None:
    label_names = preprocess_labels2.get_label_names(preprocessed_data_root_path, partition)
data_samples_file_paths = {part: preprocessed_data_root_path / f'{part}_form_labels_data_samples.pt' for part in partition}
if all([data_samples_file_paths[part].exists() for part in data_samples_file_paths]):
    for part in partition:
        file_path = data_samples_file_paths[part]
        logging.info(f'Loading {tb_schema} form labels tensor dataset to file {file_path}')
        datasets[part] = torch.load(file_path)
else:
    datasets = load_preprocessed_data_samples(preprocessed_data_root_path, partition, label_names)
    for part in datasets:
        file_path = data_samples_file_paths[part]
        logging.info(f'Saving {tb_schema} form labels tensor dataset to file {file_path}')
        torch.save(datasets[part], file_path)
# datasets['train'] = TensorDataset(*[t[:10] for t in datasets['train'].tensors])
# datasets['dev'] = TensorDataset(*[t[:100] for t in datasets['dev'].tensors])
# datasets['test'] = TensorDataset(*[t[:100] for t in datasets['test'].tensors])
train_dataloader = DataLoader(datasets['train'], batch_size=1, shuffle=False)
dev_dataloader = DataLoader(datasets['dev'], batch_size=100)
test_dataloader = DataLoader(datasets['test'], batch_size=100)

# Language Model
bert_folder_path = Path(f'./experiments/transformers/bert/{bert_model_type}/{bert_tokenizer_type}/{bert_version}')
if bert_tokenizer_type == 'roots':
    logging.info(f'Loading roots tokenizer BERT from: {str(bert_folder_path)}')
    bert_tokenizer = AlefBERTRootTokenizer(str(bert_folder_path / 'vocab.txt'))
    bert = BertModel.from_pretrained(str(bert_folder_path))
elif bert_version == 'mbert':
    logging.info(f'Loading mBERT')
    bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')
    bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
elif bert_version == 'mbert-cased':
    logging.info(f'Loading mBERT cased')
    bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
    bert = BertModel.from_pretrained('bert-base-multilingual-cased')
elif bert_version == 'hebert':
    logging.info(f'Loading heBERT')
    bert_tokenizer = BertTokenizerFast.from_pretrained('avichr/heBERT')
    bert = BertModel.from_pretrained('avichr/heBERT')
else:
    logging.info(f'Loading BERT from: {str(bert_folder_path)}')
    bert_tokenizer = BertTokenizerFast.from_pretrained(str(bert_folder_path))
    bert = BertModel.from_pretrained(str(bert_folder_path))
logging.info('BERT model and tokenizer loaded')

# MD Model
pad, sos, eos, sep = '<pad>', '<s>', '</s>', '<sep>'
char_vectors, char_vocab = preprocess_labels2.load_char_vocab(preprocessed_data_root_path)
label_vocab = preprocess_labels2.load_label_vocab(preprocessed_data_root_path, partition, pad=pad)
char_tensors = torch.tensor(char_vectors, dtype=torch.float)
char_emb_pad_id = char_vocab['char2id'][pad]
char_emb = nn.Embedding.from_pretrained(char_tensors, freeze=False, padding_idx=char_emb_pad_id)
num_labels = [len(label_vocab['labels2id'][l]) for l in label_names]
num_layers = 2
hidden_size = bert.config.hidden_size // num_layers
dropout = 0.1
num_chars = len(char_vocab['char2id'])
out_dropout = 0.5

xtoken_emb = BertTokenEmbeddingModel(bert, bert_tokenizer)
if md_strategry == "morph-pipeline":
    segmentor = SegmentDecoder(char_emb, hidden_size, num_layers, dropout, out_dropout, num_chars, [])
    md_model = MorphPipelineModel(xtoken_emb, segmentor, hidden_size, num_layers, dropout, out_dropout, num_labels)
elif md_strategry == "morph-sequence":
    segmentor = SegmentDecoder(char_emb, hidden_size, num_layers, dropout, out_dropout, num_chars, num_labels)
    md_model = MorphSequenceModel(xtoken_emb, segmentor)
else:
    segmentor = SegmentDecoder(char_emb, hidden_size, num_layers, dropout, out_dropout, num_chars, [])
    md_model = MorphSequenceModel(xtoken_emb, segmentor)
device = None
if device is not None:
    md_model.to(device)
print(md_model)

# Special symbols
char_sos = torch.tensor([char_vocab['char2id'][sos]], dtype=torch.long)
char_eos = torch.tensor([char_vocab['char2id'][eos]], dtype=torch.long)
char_sep = torch.tensor([char_vocab['char2id'][sep]], dtype=torch.long)
char_pad = torch.tensor([char_vocab['char2id'][pad]], dtype=torch.long)
char_special_symbols = {sos: char_sos.to(device), eos: char_eos.to(device),
                        sep: char_sep.to(device), pad: char_pad.to(device)}
label_pads = [torch.tensor([label_vocab['labels2id'][l][pad]], dtype=torch.long) for l in label_names]


def to_sent_tokens(token_chars) -> list:
    tokens = []
    for chars in token_chars:
        token = ''.join([char_vocab['id2char'][c] for c in chars[chars > 0].tolist()])
        tokens.append(token)
    return tokens


def to_token_morph_segments(chars) -> list:
    tokens = []
    token_mask = torch.nonzero(torch.eq(chars, char_eos))
    token_mask_map = {m[0].item(): m[1].item() for m in token_mask}
    for i, token_chars in enumerate(chars):
        token_len = token_mask_map[i] if i in token_mask_map else chars.shape[1]
        token_chars = token_chars[:token_len]
        form_mask = torch.nonzero(torch.eq(token_chars, char_sep))
        forms = []
        start_pos = 0
        for to_pos in form_mask:
            form_chars = token_chars[start_pos:to_pos.item()]
            form = ''.join([char_vocab['id2char'][c.item()] for c in form_chars])
            forms.append(form)
            start_pos = to_pos.item() + 1
        form_chars = token_chars[start_pos:]
        form = ''.join([char_vocab['id2char'][c.item()] for c in form_chars])
        forms.append(form)
        tokens.append(forms)
    return tokens


def to_token_morph_labels(labels, label_names) -> list:
    tokens = []
    for i, (feat_labels, feat_name) in enumerate(zip(labels, label_names)):
        morph_labels = []
        for token_feat_labels in feat_labels:
            token_feat_labels = token_feat_labels[token_feat_labels != label_pads[i]]
            token_feat_labels = [label_vocab['id2labels'][feat_name][t.item()] for t in token_feat_labels]
            morph_labels.append(token_feat_labels)
        tokens.append(morph_labels)
    return list(map(list, zip(*tokens)))


def to_feats_strs(labels: dict) -> list:
    feats_strs = []
    feature_names = sorted(labels)
    feature_values = [labels[feat_name] for feat_name in feature_names]
    feature_values = [f for f in itertools.zip_longest(*feature_values, fillvalue='_')]
    for fvalues in feature_values:
        feats_str = '|'.join([f'{feature_names[j]}={fvalues[j]}' for j in range(len(feature_names))])
        feats_strs.append(feats_str)
    return feats_strs


def to_sent_token_lattice_rows(sent_id, tokens, token_segments, token_labels) -> list:
    rows = []
    node_id = 0
    for token_id, (token, forms, labels) in enumerate(zip(tokens, token_segments, token_labels)):
        labels = {label_names[j]: labels[j] for j in range(len(label_names))}
        tags = labels['tag'] if 'tag' in labels else ['_' for _ in range(len(forms))]
        feats_strs = to_feats_strs({k: v for k, v in labels.items() if k != 'tag'})
        for form, tag, feat in zip(forms, tags, feats_strs):
            row = [sent_id, node_id, node_id+1, form, '_', tag, feat, token_id+1, token, True]
            rows.append(row)
            node_id += 1
    return rows


def get_lattice_data(sent_token_seg_tag_rows) -> pd.DataFrame:
    lattice_rows = []
    for row in sent_token_seg_tag_rows:
        lattice_rows.extend(to_sent_token_lattice_rows(*row))
    columns = ['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token', 'is_gold']
    return pd.DataFrame(lattice_rows, columns=columns)


def morph_eval(decoded_sent_tokens, target_sent_tokens) -> (tuple, tuple):
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


def print_eval_scores(decoded_df, truth_df, fields, step):
    aligned_scores, mset_scores = tb.morph_eval(pred_df=decoded_df, gold_df=truth_df, fields=fields)
    for fs in aligned_scores:
        p, r, f = aligned_scores[fs]
        print(f'eval {step} aligned {fs}: [P: {p}, R: {r}, F: {f}]')
        p, r, f = mset_scores[fs]
        print(f'eval {step} mset    {fs}: [P: {p}, R: {r}, F: {f}]')


def save_ner(df, out_file_path):
    gb = df.groupby('sent_id')
    with open(out_file_path, 'w') as f:
        for sid, group in gb:
            for row in group[['form', 'tag']].itertuples():
                f.write(f'{row.form} {row.tag}\n')
            f.write('\n')


# Training and evaluation routine
def process(model: MorphSequenceModel, data: DataLoader, criterion: nn.CrossEntropyLoss, epoch, phase, print_every,
            teacher_forcing_ratio=0.0, optimizer: optim.AdamW = None, max_grad_norm=None):
    print_form_loss, total_form_loss = 0, 0
    print_label_losses, total_label_losses = [0 for _ in range(len(label_names))], [0 for _ in range(len(label_names))]
    print_target_forms, total_target_forms = [], []
    print_target_labels, total_target_labels = [], []
    print_decoded_forms, total_decoded_forms = [], []
    print_decoded_labels, total_decoded_labels = [], []
    print_decoded_lattice_rows, total_decoded_lattice_rows = [], []

    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        batch_form_scores, batch_label_scores, batch_form_targets, batch_label_targets = [], [], [], []
        batch_token_chars, batch_sent_ids, batch_num_tokens = [], [], []
        for sent_xtoken, sent_token_chars, sent_form_chars, sent_labels in zip(*batch):
            input_token_chars = sent_token_chars[:, :, -1]
            num_tokens = len(sent_token_chars[sent_token_chars[:, 0, 1] > 0])
            target_token_form_chars = sent_form_chars[:, :, -1]
            max_form_len = target_token_form_chars.shape[1]
            target_token_labels = sent_labels[:, :, 2:]
            max_num_labels = target_token_labels.shape[1]
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            form_scores, _, label_scores = model(sent_xtoken, input_token_chars, char_special_symbols, num_tokens,
                                                 max_form_len, max_num_labels,
                                                 target_token_form_chars if use_teacher_forcing else None)
            batch_form_scores.append(form_scores)
            batch_label_scores.append(label_scores)
            batch_form_targets.append(target_token_form_chars[:num_tokens])
            batch_label_targets.append(target_token_labels[:num_tokens])
            batch_token_chars.append(input_token_chars[:num_tokens])
            batch_sent_ids.append(sent_form_chars[:, :, 0].unique().item())
            batch_num_tokens.append(num_tokens)

        # Decode
        batch_form_scores = nn.utils.rnn.pad_sequence(batch_form_scores, batch_first=True)
        batch_label_scores = [nn.utils.rnn.pad_sequence(label_scores, batch_first=True)
                              for label_scores in list(map(list, zip(*batch_label_scores)))]
        with torch.no_grad():
            batch_decoded_chars, batch_decoded_labels = model.decode(batch_form_scores, batch_label_scores)

        # Form Loss
        batch_form_targets = nn.utils.rnn.pad_sequence(batch_form_targets, batch_first=True)
        loss_batch_form_targets = batch_form_targets.view(-1)
        loss_batch_form_scores = batch_form_scores.view(-1, batch_form_scores.shape[-1])
        form_loss = criterion(loss_batch_form_scores, loss_batch_form_targets)
        print_form_loss += form_loss.item()

        # Label Losses
        label_losses = []
        batch_label_targets = [[t[:, :, j] for j in range(t.shape[-1])] for t in batch_label_targets]
        batch_label_targets = [nn.utils.rnn.pad_sequence(label_targets, batch_first=True)
                               for label_targets in list(map(list, zip(*batch_label_targets)))]
        for j, (label_scores, label_targets) in enumerate(zip(batch_label_scores, batch_label_targets)):
            loss_label_targets = label_targets.view(-1)
            loss_label_scores = label_scores.view(-1, label_scores.shape[-1])
            label_loss = criterion(loss_label_scores, loss_label_targets)
            label_losses.append(label_loss)
            print_label_losses[j] += label_loss.item()

        # Optimization Step
        if optimizer is not None:
            form_loss.backward(retain_graph=len(label_losses) > 0)
            for j in range(len(label_losses)):
                label_losses[j].backward(retain_graph=(j < len(label_losses)-1))
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # To Lattice
        for j in range(len(batch_sent_ids)):
            sent_id = batch_sent_ids[j]
            input_chars = batch_token_chars[j]
            target_form_chars = batch_form_targets[j]
            target_labels = [label_targets[j] for label_targets in batch_label_targets]
            decoded_form_chars = batch_decoded_chars[j]
            decoded_labels = [decoded_labels[j] for decoded_labels in batch_decoded_labels]
            num_tokens = batch_num_tokens[j]
            input_chars = input_chars.to('cpu')
            target_form_chars = target_form_chars[:num_tokens].to('cpu')
            decoded_form_chars = decoded_form_chars[:num_tokens].to('cpu')
            target_labels = [labels[:num_tokens].to('cpu') for labels in target_labels]
            decoded_labels = [labels[:num_tokens].to('cpu') for labels in decoded_labels]
            input_tokens = to_sent_tokens(input_chars)
            target_morph_segments = to_token_morph_segments(target_form_chars)
            decoded_morph_segments = to_token_morph_segments(decoded_form_chars)
            target_morph_labels = to_token_morph_labels(target_labels, label_names)
            decoded_morph_labels = to_token_morph_labels(decoded_labels, label_names)

            decoded_token_lattice_rows = (sent_id, input_tokens, decoded_morph_segments, decoded_morph_labels)
            print_decoded_lattice_rows.append(decoded_token_lattice_rows)
            print_target_forms.append(target_morph_segments)
            print_target_labels.append(target_morph_labels)
            print_decoded_forms.append(decoded_morph_segments)
            print_decoded_labels.append(decoded_morph_labels)

        # Log Print Eval
        if (i + 1) % print_every == 0:
            sent_id, input_tokens, decoded_segments, decoded_labels = print_decoded_lattice_rows[-1]
            target_segments = print_target_forms[-1]
            target_labels = print_target_labels[-1]
            decoded_segments = print_decoded_forms[-1]
            decoded_labels = print_decoded_labels[-1]

            print(f'epoch {epoch} {phase}, step {i + 1} form char loss: {print_form_loss / print_every}')
            for j in range(len(label_names)):
                print(f'epoch {epoch} {phase}, step {i + 1} {label_names[j]} loss: {print_label_losses[j] / print_every}')
            print(f'sent #{sent_id} input tokens  : {input_tokens}')
            print(f'sent #{sent_id} target forms  : {list(reversed(target_segments))}')
            print(f'sent #{sent_id} decoded forms : {list(reversed(decoded_segments))}')
            for j in range(len(label_names)):
                label_values = [labels[j] for labels in target_labels]
                print(f'sent #{sent_id} target {label_names[j]} labels  : {list(reversed([label_values]))}')
                label_values = [labels[j] for labels in decoded_labels]
                print(f'sent #{sent_id} decoded {label_names[j]} labels : {list(reversed([label_values]))}')
            total_form_loss += print_form_loss
            for j, label_loss in enumerate(print_label_losses):
                total_label_losses[j] += label_loss
            print_form_loss = 0
            print_label_losses = [0 for _ in range(len(label_names))]

            total_decoded_forms.extend(print_decoded_forms)
            total_decoded_labels.extend(print_decoded_labels)
            total_target_forms.extend(print_target_forms)
            total_target_labels.extend(print_target_labels)
            total_decoded_lattice_rows.extend(print_decoded_lattice_rows)

            aligned_scores, mset_scores = morph_eval(print_decoded_forms, print_target_forms)
            print(f'form aligned scores: {aligned_scores}')
            print(f'form mset scores: {mset_scores}')

            for j in range(len(label_names)):
                decoded_values = [labels[j] for sent_labels in print_decoded_labels for labels in sent_labels]
                target_values = [labels[j] for sent_labels in print_target_labels for labels in sent_labels]
                aligned_scores, mset_scores = morph_eval(decoded_values, target_values)
                print(f'{label_names[j]} aligned scores: {aligned_scores}')
                print(f'{label_names[j]} mset scores: {mset_scores}')

            print_target_forms = []
            print_target_labels = []
            print_decoded_forms = []
            print_decoded_labels = []
            print_decoded_lattice_rows = []

    # Log Total Eval
    if print_form_loss > 0:
        total_form_loss += print_form_loss
        for j, label_loss in enumerate(print_label_losses):
            total_label_losses[j] += label_loss
        total_decoded_forms.extend(print_decoded_forms)
        total_decoded_labels.extend(print_decoded_labels)
        total_target_forms.extend(print_target_forms)
        total_target_labels.extend(print_target_labels)
        total_decoded_lattice_rows.extend(print_decoded_lattice_rows)
    print(f'epoch {epoch} {phase}, total form char loss: {total_form_loss / len(data)}')
    for j in range(len(label_names)):
        print(f'epoch {epoch} {phase}, total {label_names[j]} loss: {total_label_losses[j] / len(data)}')

    aligned_scores, mset_scores = morph_eval(total_decoded_forms, total_target_forms)
    print(f'form total aligned scores: {aligned_scores}')
    print(f'form total mset scores: {mset_scores}')

    total_decoded_labels = list(map(list, zip(*total_decoded_labels)))
    total_target_labels = list(map(list, zip(*total_target_labels)))
    for j in range(len(label_names)):
        decoded_values = [labels[j] for sent_labels in total_decoded_labels for labels in sent_labels]
        target_values = [labels[j] for sent_labels in total_target_labels for labels in sent_labels]
        aligned_scores, mset_scores = morph_eval(decoded_values, target_values)
        print(f'{label_names[j]} total aligned scores: {aligned_scores}')
        print(f'{label_names[j]} total mset scores: {mset_scores}')

    return get_lattice_data(total_decoded_lattice_rows)


# Optimizer
epochs = 3
max_grad_norm = 1.0
lr = 1e-3
# freeze bert
for param in bert.parameters():
    param.requires_grad = False
parameters = list(filter(lambda p: p.requires_grad, md_model.parameters()))
# parameters = morph_tagger_model.parameters()
adam = optim.AdamW(parameters, lr=lr)
loss_fct = nn.CrossEntropyLoss(ignore_index=0)
teacher_forcing_ratio = 1.0

out_path = Path(f'experiments/morph-seg-ner/bert/distilled/wordpiece/{bert_version}/UD_Hebrew/HTB')
out_path.mkdir(parents=True, exist_ok=True)
eval_fields = ['form']
if 'tag' in label_names:
    eval_fields.append('tag')
if len([name for name in label_names if name not in ['ner', 'tag']]) > 0:
    eval_fields.append('feats')


# Training epochs
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    md_model.train()
    process(md_model, train_dataloader, loss_fct, epoch, 'train', 10, teacher_forcing_ratio, adam, max_grad_norm)
    md_model.eval()
    with torch.no_grad():
        dev_samples = process(md_model, dev_dataloader, loss_fct, epoch, 'dev', 1)
        print_eval_scores(decoded_df=dev_samples, truth_df=partition['dev'], step=epoch, fields=eval_fields)
        test_samples = process(md_model, test_dataloader, loss_fct, epoch, 'test', 1)
        print_eval_scores(decoded_df=test_samples, truth_df=partition['test'], step=epoch, fields=eval_fields)

        if 'ner' in label_names:
            save_ner(dev_samples, out_path / 'morph_label_dev.bmes')
            dev_gold_file_path = 'data/raw/for_amit_spmrl/hebtb/gold/morph_gold_dev.bmes'
            print(ne_evaluate_mentions.evaluate_files(dev_gold_file_path, out_path / 'morph_label_dev.bmes'))
            print(ne_evaluate_mentions.evaluate_files(dev_gold_file_path, out_path / 'morph_label_dev.bmes',
                                                      ignore_cat=True))

            save_ner(test_samples, out_path / 'morph_label_test.bmes')
            test_gold_file_path = 'data/raw/for_amit_spmrl/hebtb/gold/morph_gold_test.bmes'
            print(ne_evaluate_mentions.evaluate_files(test_gold_file_path, out_path / 'morph_label_test.bmes'))
            print(ne_evaluate_mentions.evaluate_files(test_gold_file_path, out_path / 'morph_label_test.bmes',
                                                      ignore_cat=True))
