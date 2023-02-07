import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb as wandb
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from transformers import AutoModel, AutoTokenizer

import utils
from bclm import treebank as tb
from constants import PAD, SOS, EOS, SEP
from data import preprocess_form, preprocess_labels
from morph_model import BertTokenEmbeddingModel, SegmentDecoder, MorphSequenceModel, MorphPipelineModel
from utils import freeze_model, unfreeze_model, get_ud_preprocessed_dir_path


def main(config):
    wandb.init(project='md_model', config=config)

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # Config
    # treebank schema (UD)
    tb_schema = config['tb_schema']
    # language/treebank
    lang = config['lang']
    tb_name = config['tb_name']
    la_name = config['la_name']

    # bert
    bert_tokenizer_name = config['bert_tokenizer_name']
    bert_tokenizer_path = config['bert_tokenizer_path']
    tokenizer_type = config.get('tokenizer_type', 'auto')
    bert_model_name = config['bert_model_name']
    bert_model_path = config['bert_model_path']
    model_type = config.get('model_type', 'auto')

    # # bert params
    # bert_vocab_size = config['bert_vocab_size']
    # bert_epochs = config['bert_epochs']
    # bert_corpus_name = config['bert_corpus_name']
    # bert_model_size_type = config['bert_model_size_type']

    # paths
    raw_root_path = Path(config['raw_root_path'])
    preprocessed_root_path = Path(config['preprocessed_root_path'])
    preprocessed_dir_path = get_ud_preprocessed_dir_path(preprocessed_root_path, lang, tb_name, bert_tokenizer_name)
    out_root_path = Path(config['out_root_path'])

    # control
    do_train = config.get('do_train', False)
    predict_train = config.get('predict_train', False)
    do_eval = config.get('do_eval', False)
    do_test = config.get('do_test', False)

    # morph model
    md_strategry = config['md_strategry']
    label_names = config['label_names']
    use_crf_for_ner = config.get('use_crf_for_ner', False)

    # train
    device = config['device']
    epochs = config['epochs']
    epochs_frozen = config['epochs_frozen']
    eval_epochs = config['eval_epochs']
    model_checkpoint_path = config.get('model_checkpoint_path', None)
    checkpoint_epochs = config.get('checkpoint_epochs', 0)
    lr = config.get('lr', 1e-3)
    lr_scheduler_cls_name = config['lr_scheduler_cls_name']
    lr_scheduler_params = config['lr_scheduler_params']

    mask_extra_tokens_label = config.get('mask_extra_tokens_label', None)
    mask_extra_tokens_value = config.get('mask_extra_tokens_value', None)
    eval_fix_extra_tokens = mask_extra_tokens_label is not None

    # data
    ner_feat_name = config.get('ner_feat_name', 'ner')

    if tb_schema == 'UD':
        partition = tb.ud(raw_root_path,
                          tb_root_path=raw_root_path,
                          lang=lang,
                          tb_name=tb_name,
                          la_name=la_name)
    else:
        partition = {'train': None, 'dev': None, 'test': None}

    datasets = {}

    if label_names is None:
        label_names = preprocess_labels.get_label_names(preprocessed_dir_path, partition)

    # Output folder path
    out_morph_type = 'morph_seg'
    if len(label_names) == 0:
        pass
    elif len(label_names) == 1:
        out_morph_type = f'{out_morph_type}_{label_names[0]}'
    elif len(label_names) == 2:
        out_morph_type = f'{out_morph_type}_{label_names[0]}_{label_names[1]}'
    else:
        if 'tag' in label_names:
            out_morph_type = f'{out_morph_type}_tag_feats'
        else:
            out_morph_type = f'{out_morph_type}_feats'

    out_dir_path = out_root_path / lang / tb_name / out_morph_type / bert_model_name / bert_tokenizer_name
    out_dir_path.mkdir(parents=True, exist_ok=True)

    with open(out_dir_path / "config.json", 'w') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    data_samples_file_paths = {part: preprocessed_dir_path / f'{part}_{out_morph_type}_data_samples.pt'
                               for part in partition}
    if all([data_samples_file_paths[part].exists() for part in data_samples_file_paths]):
        for part in partition:
            file_path = data_samples_file_paths[part]
            logging.info(f'Loading {tb_schema} {out_morph_type} tensor dataset from {file_path}')
            datasets[part] = torch.load(file_path)
    else:
        datasets = load_preprocessed_data_samples(preprocessed_dir_path, partition, label_names)
        for part in datasets:
            file_path = data_samples_file_paths[part]
            logging.info(f'Saving {tb_schema} {out_morph_type} tensor dataset to {file_path}')
            torch.save(datasets[part], file_path)

    train_dataloader = DataLoader(datasets['train'], batch_size=4, shuffle=False)
    dev_dataloader = DataLoader(datasets['dev'], batch_size=100)
    test_dataloader = DataLoader(datasets['test'], batch_size=100)

    # Language Model
    tokenizer = utils.get_tokenizer(tokenizer_type, bert_tokenizer_path)
    bert = utils.get_model(model_type, bert_model_path)

    # Vocabs
    char_vectors, char_vocab = preprocess_labels.load_char_vocab(preprocessed_dir_path)
    label_vocab = preprocess_labels.load_label_vocab(preprocessed_dir_path, partition, pad=PAD)

    # Special symbols
    char_sos = torch.tensor([char_vocab['char2id'][SOS]], dtype=torch.long)
    char_eos = torch.tensor([char_vocab['char2id'][EOS]], dtype=torch.long)
    char_sep = torch.tensor([char_vocab['char2id'][SEP]], dtype=torch.long)
    char_pad = torch.tensor([char_vocab['char2id'][PAD]], dtype=torch.long)
    label_pads = [torch.tensor([label_vocab['labels2id'][l][PAD]], dtype=torch.long) for l in label_names]

    # MD Model
    char_tensors = torch.tensor(char_vectors, dtype=torch.float)
    char_emb_pad_id = char_vocab['char2id'][PAD]
    char_emb = nn.Embedding.from_pretrained(char_tensors, freeze=False, padding_idx=char_emb_pad_id)
    # num_labels = {len(label_vocab['labels2id'][name]) for name in label_names}
    num_layers = 2
    hidden_size = bert.config.hidden_size // num_layers
    dropout = 0.1
    num_chars = len(char_vocab['char2id'])
    out_dropout = 0.5
    xtoken_emb = BertTokenEmbeddingModel(bert, tokenizer)
    label_classifier_configs = []
    for name in label_names:
        label_classifier_config = {'id2label': label_vocab['id2labels'][name]}
        if use_crf_for_ner and name == ner_feat_name:
            label_classifier_config['crf_trans_type'] = 'BIOSE'
        label_classifier_configs.append(label_classifier_config)

    if md_strategry == "morph-pipeline":
        segmentor = SegmentDecoder(char_emb, hidden_size, num_layers, dropout, out_dropout, num_chars)
        md_model = MorphPipelineModel(xtoken_emb, segmentor, hidden_size, num_layers, dropout, out_dropout,
                                      label_classifier_configs)
    elif md_strategry == "morph-sequence":
        segmentor = SegmentDecoder(char_emb, hidden_size, num_layers, dropout, out_dropout, num_chars,
                                   label_classifier_configs)
        md_model = MorphSequenceModel(xtoken_emb, segmentor)
    elif md_strategry == "segment-only":
        segmentor = SegmentDecoder(char_emb, hidden_size, num_layers, dropout, out_dropout, num_chars)
        md_model = MorphSequenceModel(xtoken_emb, segmentor)
    else:
        raise KeyError(f'unknown md_strategry {md_strategry}')

    # load from checkpoint
    if model_checkpoint_path:
        print(f'loading model from checkpoint {model_checkpoint_path}')

        with open(model_checkpoint_path, 'rb') as f:
            state_dict = torch.load(f)
        md_model.load_state_dict(state_dict)

    # device
    if device == 'auto':
        device, _ = utils.get_most_free_device()

    char_special_symbols = {SOS: char_sos, EOS: char_eos,
                            SEP: char_sep, PAD: char_pad}
    if device is not None:
        char_special_symbols = {k: v.to(device) for k, v in char_special_symbols.items()}
        md_model.to(device)
    print(md_model)

    eval_fields = ['form']
    if 'tag' in label_names:
        eval_fields.append('tag')
    # feats (e.g. Gender, Number etc.)
    keep_feats = None
    additional_fields = [name for name in label_names if name not in [ner_feat_name, 'tag']]

    if len(additional_fields) > 0:
        eval_fields.append('feats')
        keep_feats = additional_fields[:]

    # Optimizer
    max_grad_norm = 1.0

    # freeze bert
    if epochs_frozen > 0 and (checkpoint_epochs is not None and epochs_frozen > checkpoint_epochs):
        print("freezing bert")
        freeze_model(bert)

    parameters = list(filter(lambda p: p.requires_grad, md_model.parameters()))
    # parameters = morph_tagger_model.parameters()
    optimizer = optim.AdamW(parameters, lr=lr)

    lr_scheduler_cls = getattr(torch.optim.lr_scheduler, lr_scheduler_cls_name)
    lr_scheduler = lr_scheduler_cls(optimizer, **lr_scheduler_params)

    loss_fct = nn.CrossEntropyLoss(ignore_index=0)
    teacher_forcing_ratio = 1.0
    # don't print
    print_every = None

    epoch_offset = 1
    if checkpoint_epochs:
        epoch_offset += checkpoint_epochs
        for _ in range(checkpoint_epochs):
            lr_scheduler.step()

    if do_train:
        # Training epochs
        for i in trange(epochs - checkpoint_epochs, desc="Epoch"):
            epoch = epoch_offset + i
            print('epoch', epoch, 'lr', lr_scheduler.get_last_lr())

            out_epoch_dir_path = out_dir_path / str(epoch)
            out_epoch_dir_path.mkdir(parents=True, exist_ok=True)

            # unfreeze
            if 0 < epochs_frozen == epoch - 1:
                print("unfreezing bert")

                unfreeze_model(bert)

                # recreate optimizer and lr_scheduler with new parameters
                parameters = list(filter(lambda p: p.requires_grad, md_model.parameters()))
                optimizer = optim.AdamW(parameters, lr=lr)
                lr_scheduler = lr_scheduler_cls(optimizer, **lr_scheduler_params)
                for _ in range(epochs_frozen):
                    lr_scheduler.step()

            # train
            md_model.train()
            process(md_model, train_dataloader, label_names, char_vocab, label_vocab, char_special_symbols, label_pads,
                    loss_fct, epoch, 'train', print_every, teacher_forcing_ratio=teacher_forcing_ratio, device=device,
                    optimizer=optimizer, max_grad_norm=max_grad_norm,
                    mask_extra_tokens_label=mask_extra_tokens_label,
                    mask_extra_tokens_value=mask_extra_tokens_value)
            lr_scheduler.step()

            # save model
            torch.save(md_model.state_dict(), out_epoch_dir_path / "md_model.pt")

            # eval
            if epoch % eval_epochs == 0:
                run_all_predict(predict_train, do_eval, do_test, char_special_symbols, char_vocab, dev_dataloader,
                                device, epoch, eval_fields, eval_fix_extra_tokens, label_names, label_pads, label_vocab,
                                loss_fct, md_model, out_epoch_dir_path, partition, print_every, test_dataloader,
                                train_dataloader)

                # if 'biose_layer0' in label_names:
                #     utils.save_ner(dev_samples, out_dir_path / 'morph_label_dev.bmes', 'biose_layer0')
                #     dev_gold_file_path = root_path / Path(f'data/raw/{tb_data_src}/{tb_name}/gold/morph_gold_dev.bmes')
                #     dev_pred_file_path = out_dir_path / 'morph_label_dev.bmes'
                #     print(ne_evaluate_mentions.evaluate_files(dev_gold_file_path, dev_pred_file_path))
                #     print(ne_evaluate_mentions.evaluate_files(dev_gold_file_path, dev_pred_file_path, ignore_cat=True))
                #
                #     utils.save_ner(test_samples, out_dir_path / 'morph_label_test.bmes', 'biose_layer0')
                #     test_gold_file_path = root_path / Path(f'data/raw/{tb_data_src}/{tb_name}/gold/morph_gold_test.bmes')
                #     test_pred_file_path = out_dir_path / 'morph_label_test.bmes'
                #     print(ne_evaluate_mentions.evaluate_files(test_gold_file_path, test_pred_file_path))
                #     print(ne_evaluate_mentions.evaluate_files(test_gold_file_path, test_pred_file_path, ignore_cat=True))

    else:
        epoch = epochs
        out_epoch_dir_path = out_dir_path / str(epoch)
        out_epoch_dir_path.mkdir(parents=True, exist_ok=True)

        run_all_predict(predict_train, do_eval, do_test, char_special_symbols, char_vocab, dev_dataloader, device,
                        epoch, eval_fields, eval_fix_extra_tokens, label_names, label_pads, label_vocab, loss_fct,
                        md_model, out_epoch_dir_path, partition, print_every, test_dataloader, train_dataloader)


def run_all_predict(predict_train, do_eval, do_test, char_special_symbols, char_vocab, dev_dataloader, device, epoch,
                    eval_fields, eval_fix_extra_tokens, label_names, label_pads, label_vocab, loss_fct, md_model,
                    out_epoch_dir_path, partition, print_every, test_dataloader, train_dataloader):
    if predict_train:
        run_eval_train(char_special_symbols, char_vocab, device, epoch, eval_fields, label_names, label_pads,
                       label_vocab, loss_fct, md_model, out_epoch_dir_path, partition, print_every,
                       train_dataloader,
                       fix_extra_tokens=eval_fix_extra_tokens)
    if do_eval:
        run_eval(char_special_symbols, char_vocab, device, epoch, eval_fields, label_names, label_pads, label_vocab,
                 loss_fct, md_model, out_epoch_dir_path, partition, print_every, dev_dataloader,
                 fix_extra_tokens=eval_fix_extra_tokens)
    if do_test:
        run_test(char_special_symbols, char_vocab, device, epoch, eval_fields, label_names, label_pads,
                 label_vocab, loss_fct, md_model, out_epoch_dir_path, partition, print_every, test_dataloader,
                 fix_extra_tokens=eval_fix_extra_tokens)


def run_eval_train(char_special_symbols, char_vocab, device, epoch, eval_fields, label_names, label_pads, label_vocab,
                   loss_fct, md_model, out_epoch_dir_path, partition, print_every, train_dataloader,
                   fix_extra_tokens=False, keep_feats=None):
    md_model.eval()
    with torch.no_grad():
        train_samples = process(md_model, train_dataloader, label_names, char_vocab, label_vocab, char_special_symbols,
                                label_pads, loss_fct, epoch, 'eval_train', print_every, device=device)
        train_samples.to_csv(out_epoch_dir_path / 'train_samples.csv')
        log_dict = utils.get_wandb_log_eval_scores(decoded_df=train_samples,
                                                   truth_df=partition['train'],
                                                   phase='eval_train',
                                                   step=epoch,
                                                   fields=eval_fields,
                                                   fix_extra_tokens=fix_extra_tokens,
                                                   keep_feats=keep_feats)
        wandb.log(log_dict)


def run_eval(char_special_symbols, char_vocab, device, epoch, eval_fields, label_names, label_pads, label_vocab,
             loss_fct, md_model, out_epoch_dir_path, partition, print_every, dev_dataloader,
             fix_extra_tokens=False, keep_feats=None):
    md_model.eval()
    with torch.no_grad():
        dev_samples = process(md_model, dev_dataloader, label_names, char_vocab, label_vocab, char_special_symbols,
                              label_pads, loss_fct, epoch, 'dev', print_every, device=device)
        dev_samples.to_csv(out_epoch_dir_path / 'dev_samples.csv')
        log_dict = utils.get_wandb_log_eval_scores(decoded_df=dev_samples,
                                                   truth_df=partition['dev'],
                                                   phase='dev',
                                                   step=epoch,
                                                   fields=eval_fields,
                                                   fix_extra_tokens=fix_extra_tokens,
                                                   keep_feats=keep_feats)
        wandb.log(log_dict)


def run_test(char_special_symbols, char_vocab, device, epoch, eval_fields, label_names, label_pads, label_vocab,
             loss_fct, md_model, out_epoch_dir_path, partition, print_every, test_dataloader,
             fix_extra_tokens=False, keep_feats=None):
    md_model.eval()
    with torch.no_grad():
        test_samples = process(md_model, test_dataloader, label_names, char_vocab, label_vocab, char_special_symbols,
                               label_pads, loss_fct, epoch, 'test', print_every, device=device)
        test_samples.to_csv(out_epoch_dir_path / 'test_samples.csv')
        log_dict = utils.get_wandb_log_eval_scores(decoded_df=test_samples,
                                                   truth_df=partition['test'],
                                                   phase='test', step=epoch,
                                                   fields=eval_fields,
                                                   fix_extra_tokens=fix_extra_tokens,
                                                   keep_feats=keep_feats)
        wandb.log(log_dict)


def load_preprocessed_data_samples(data_root_path, partition, label_names) -> dict:
    logging.info(f'Loading preprocesssed {data_root_path} form tag data samples')
    xtoken_data = preprocess_form.load_xtoken_data(data_root_path, partition)
    token_char_data = preprocess_form.load_token_char_data(data_root_path, partition)
    form_char_data = preprocess_form.load_form_data(data_root_path, partition)
    label_data = preprocess_labels.load_labeled_data(data_root_path, partition, label_names=label_names)
    datasets = {}
    for part in partition:
        xtoken_tensor = torch.tensor(xtoken_data[part][:, :, 1:], dtype=torch.long)
        token_char_tensor = torch.tensor(token_char_data[part][:, :, :, 1:], dtype=torch.long)
        form_char_tensor = torch.tensor(form_char_data[part], dtype=torch.long)
        label_tensor = torch.tensor(label_data[part], dtype=torch.long)
        datasets[part] = TensorDataset(xtoken_tensor, token_char_tensor, form_char_tensor, label_tensor)
    return datasets


# Training and evaluation routine
def process(model: MorphSequenceModel, data: DataLoader, label_names: List[str], char_vocab: Dict, label_vocab: Dict,
            char_special_symbols: Dict, label_pads, criterion: nn.CrossEntropyLoss, epoch, phase, print_every,
            teacher_forcing_ratio=0.0, device='cpu', optimizer: optim.AdamW = None, max_grad_norm=None,
            mask_extra_tokens_label=None, mask_extra_tokens_value=None):
    print_form_loss, total_form_loss = 0, 0
    print_label_losses, total_label_losses = [0 for _ in range(len(label_names))], [0 for _ in range(len(label_names))]
    print_target_forms, total_target_forms = [], []
    print_target_labels, total_target_labels = [], []
    print_decoded_forms, total_decoded_forms = [], []
    print_decoded_labels, total_decoded_labels = [], []
    print_decoded_lattice_rows, total_decoded_lattice_rows = [], []

    char_special_symbols_cpu = {k: v.to('cpu') for k, v in char_special_symbols.items()}

    if mask_extra_tokens_label:
        mask_extra_tokens_target_label = label_vocab['labels2id'][mask_extra_tokens_label][mask_extra_tokens_value]
        mask_extra_tokens_label_idx = label_names.index(mask_extra_tokens_label)

    for i, batch in tqdm.tqdm(enumerate(data), desc=f"{phase}/batch"):
        batch = tuple(t.to(device) for t in batch)
        batch_form_scores = []
        batch_label_scores = []

        # prepare batch
        batch_form_targets = []
        batch_input_token_chars = []
        batch_label_targets = []
        batch_max_form_len = []
        batch_max_num_labels = []
        batch_num_tokens = []
        batch_sent_ids = []
        batch_sent_xtoken = []
        batch_target_token_form_chars = []
        batch_token_chars = []

        raw_batch_sent_xtoken, raw_batch_sent_token_chars, raw_batch_sent_form_chars, raw_batch_sent_labels = batch
        for sent_xtoken, sent_token_chars, sent_form_chars, sent_labels in zip(raw_batch_sent_xtoken,
                                                                               raw_batch_sent_token_chars,
                                                                               raw_batch_sent_form_chars,
                                                                               raw_batch_sent_labels):
            input_token_chars = sent_token_chars[:, :, -1]
            num_tokens = len(sent_token_chars[sent_token_chars[:, 0, 1] > 0])
            target_token_form_chars = sent_form_chars[:, :, -1]
            max_form_len = target_token_form_chars.shape[1]
            target_token_labels = sent_labels[:, :, 2:]
            max_num_labels = target_token_labels.shape[1]
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            form_targets = target_token_form_chars[:num_tokens]
            label_targets = target_token_labels[:num_tokens]

            # mask out labels of extra tokens (used to get for better context)
            if mask_extra_tokens_label:
                mask_extra_tokens = target_token_labels[:num_tokens, 0,
                                    mask_extra_tokens_label_idx] != mask_extra_tokens_target_label

                form_targets *= mask_extra_tokens.view(-1, 1)
                label_targets *= mask_extra_tokens.view(-1, 1, 1)

            batch_form_targets.append(form_targets)
            batch_input_token_chars.append(input_token_chars)
            batch_label_targets.append(label_targets)
            batch_max_form_len.append(max_form_len)
            batch_max_num_labels.append(max_num_labels)
            batch_num_tokens.append(num_tokens)
            batch_sent_ids.append(sent_form_chars[:, :, 0].unique().item())
            batch_sent_xtoken.append(sent_xtoken)
            batch_target_token_form_chars.append(target_token_form_chars if use_teacher_forcing else None)
            batch_token_chars.append(input_token_chars[:num_tokens])

        # process batch
        for form_scores, _, label_scores in model.batch_forward(batch_sent_xtoken,
                                                                batch_input_token_chars,
                                                                char_special_symbols,
                                                                batch_num_tokens,
                                                                batch_max_form_len,
                                                                batch_max_num_labels,
                                                                batch_target_token_form_chars):
            batch_form_scores.append(form_scores)
            batch_label_scores.append(label_scores)

        # Decode
        batch_form_scores = nn.utils.rnn.pad_sequence(batch_form_scores, batch_first=True)
        batch_label_scores = [nn.utils.rnn.pad_sequence(label_scores, batch_first=True)
                              for label_scores in list(map(list, zip(*batch_label_scores)))]
        with torch.no_grad():
            batch_decoded_chars, batch_decoded_labels = model.decode(batch_form_scores, batch_label_scores)

        # Form Loss
        batch_form_targets = nn.utils.rnn.pad_sequence(batch_form_targets, batch_first=True)
        form_loss = model.form_loss(batch_form_scores, batch_form_targets, criterion)
        print_form_loss += form_loss.item()

        # Label Losses
        batch_label_targets = [[t[:, :, j] for j in range(t.shape[-1])] for t in batch_label_targets]
        batch_label_targets = [nn.utils.rnn.pad_sequence(label_targets, batch_first=True)
                               for label_targets in list(map(list, zip(*batch_label_targets)))]
        label_losses = model.labels_losses(batch_label_scores, batch_label_targets, criterion)
        for j in range(len(label_losses)):
            print_label_losses[j] += label_losses[j].item()

        # Optimization Step
        if optimizer is not None:
            mtl_loss = form_loss + sum(label_losses)
            mtl_loss.backward()
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
            input_tokens = utils.to_sent_tokens(input_chars, char_vocab['id2char'])
            target_morph_segments = utils.to_token_morph_segments(target_form_chars,
                                                                  char_vocab['id2char'],
                                                                  char_special_symbols_cpu[EOS],
                                                                  char_special_symbols_cpu[SEP])
            decoded_morph_segments = utils.to_token_morph_segments(decoded_form_chars,
                                                                   char_vocab['id2char'],
                                                                   char_special_symbols_cpu[EOS],
                                                                   char_special_symbols_cpu[SEP])
            target_morph_labels = utils.to_token_morph_labels(target_labels, label_names,
                                                              label_vocab['id2labels'],
                                                              label_pads)
            decoded_morph_labels = utils.to_token_morph_labels(decoded_labels, label_names,
                                                               label_vocab['id2labels'],
                                                               label_pads)

            decoded_token_lattice_rows = (sent_id, input_tokens, decoded_morph_segments, decoded_morph_labels)
            print_decoded_lattice_rows.append(decoded_token_lattice_rows)
            print_target_forms.append(target_morph_segments)
            print_target_labels.append(target_morph_labels)
            print_decoded_forms.append(decoded_morph_segments)
            print_decoded_labels.append(decoded_morph_labels)

        # Log Print Eval
        if print_every is not None and (i + 1) % print_every == 0:
            sent_id, input_tokens, decoded_segments, decoded_labels = print_decoded_lattice_rows[-1]
            target_segments = print_target_forms[-1]
            target_labels = print_target_labels[-1]
            decoded_segments = print_decoded_forms[-1]
            decoded_labels = print_decoded_labels[-1]

            print(f'epoch {epoch} {phase}, batch {i + 1} form char loss: {print_form_loss / print_every}')
            for j in range(len(label_names)):
                print(
                    f'epoch {epoch} {phase}, batch {i + 1} {label_names[j]} loss: {print_label_losses[j] / print_every}')
            print(f'epoch {epoch} {phase}, batch {i + 1} sent #{sent_id} input tokens  : {input_tokens}')
            print(
                f'epoch {epoch} {phase}, batch {i + 1} sent #{sent_id} target forms  : {list(reversed(target_segments))}')
            print(
                f'epoch {epoch} {phase}, batch {i + 1} sent #{sent_id} decoded forms : {list(reversed(decoded_segments))}')
            for j in range(len(label_names)):
                target_values = [labels[j] for labels in target_labels]
                print(
                    f'epoch {epoch} {phase}, batch {i + 1} sent #{sent_id} target {label_names[j]} labels  : {list(reversed([target_values]))}')
                decoded_values = [labels[j] for labels in decoded_labels]
                print(
                    f'epoch {epoch} {phase}, batch {i + 1} sent #{sent_id} decoded {label_names[j]} labels : {list(reversed([decoded_values]))}')
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

            aligned_scores, mset_scores = utils.morph_eval(print_decoded_forms, print_target_forms)
            # print(f'epoch {epoch} {phase}, batch {i + 1} form aligned scores: {aligned_scores}')
            print(f'epoch {epoch} {phase}, batch {i + 1} form mset scores: {mset_scores}')

            for j in range(len(label_names)):
                if label_names[j][:3].lower() in ['tag', 'bio', 'gen', 'num', 'per', 'ten']:
                    decoded_values = [labels[j] for sent_labels in print_decoded_labels for labels in sent_labels]
                    target_values = [labels[j] for sent_labels in print_target_labels for labels in sent_labels]
                    aligned_scores, mset_scores = utils.morph_eval(decoded_values, target_values)
                    # print(f'epoch {epoch} {phase}, batch {i + 1} {label_names[j]} aligned scores: {aligned_scores}')
                    print(f'epoch {epoch} {phase}, batch {i + 1} {label_names[j]} mset scores: {mset_scores}')

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

    log_dict = {
        'epoch': epoch,
        'phase': phase,
        'form_loss': total_form_loss / len(data),
    }

    for j in range(len(label_names)):
        log_dict[f'{label_names[j]}_loss'] = total_label_losses[j] / len(data)

    for j in range(len(label_names)):
        if label_names[j][:3].lower() in ['tag', 'bio', 'gen', 'num', 'per', 'ten']:
            decoded_values = [labels[j] for sent_labels in total_decoded_labels for labels in sent_labels]
            target_values = [labels[j] for sent_labels in total_target_labels for labels in sent_labels]
            aligned_scores, mset_scores = utils.morph_eval(decoded_values, target_values)
            log_dict[f'{label_names[j]}_aligned_p'] = aligned_scores[0]
            log_dict[f'{label_names[j]}_aligned_r'] = aligned_scores[1]
            log_dict[f'{label_names[j]}_aligned_f1'] = aligned_scores[2]
            log_dict[f'{label_names[j]}_mset_p'] = mset_scores[0]
            log_dict[f'{label_names[j]}_mset_r'] = mset_scores[1]
            log_dict[f'{label_names[j]}_mset_f1'] = mset_scores[2]

    wandb.log(log_dict)

    return utils.get_lattice_data(total_decoded_lattice_rows, label_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=Path, help='config path')
    args = parser.parse_args()

    config_path = args.config_path

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    logging.info(config)

    main(config)
