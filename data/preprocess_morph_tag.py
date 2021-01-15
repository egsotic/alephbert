from .preprocess_base import *


def _collate_morph_tag_data_samples(morph_df: pd.DataFrame, tag2index: dict, include_eos: bool):
    sent_groups = morph_df.groupby([morph_df.sent_id])
    num_sentences = len(sent_groups)
    token_groups = sorted(morph_df.groupby([morph_df.sent_id, morph_df.token_id]))
    max_num_morphemes = max([len(token_df) for _, token_df in token_groups])
    if include_eos:
        max_num_morphemes += 1
    data_sent_idx, data_token_idx, data_tokens, data_morph_idx = [], [], [], []
    data_forms, data_tags, data_tag_ids = [], [], []
    prev_sent_id = 1
    tq = tqdm(total=num_sentences, desc="Sentence")
    for (sent_id, token_id), token_df in token_groups:
        sent_idxs = list(token_df.sent_id)
        token_idxs = list(token_df.token_id)
        tokens = list(token_df.token)
        morph_idxs = list(token_df.morph_id)
        forms = list(token_df.form)
        tags = list(token_df.tag)
        tag_ids = [tag2index[t] for t in tags]
        if include_eos:
            sent_idxs += sent_idxs[-1:]
            token_idxs += token_idxs[-1:]
            tokens += tokens[-1:]
            morph_idxs += [-1]
            forms += forms[-1:]
            tags += ['</s>']
            tag_ids += [tag2index['</s>']]

        pad_len = max_num_morphemes - len(tags)
        sent_idxs.extend(sent_idxs[-1:] * pad_len)
        token_idxs.extend(token_idxs[-1:] * pad_len)
        tokens.extend(tokens[-1:] * pad_len)
        morph_idxs.extend([-1] * pad_len)
        forms.extend(['<pad>'] * pad_len)
        tags.extend(['<pad>'] * pad_len)
        tag_ids.extend([tag2index['<pad>']] * pad_len)

        if sent_id != prev_sent_id:
            tq.update(1)
            prev_sent_id = sent_id

        data_sent_idx.extend(sent_idxs)
        data_token_idx.extend(token_idxs)
        data_tokens.extend(tokens)
        data_morph_idx.extend(morph_idxs)
        data_forms.extend(forms)
        data_tags.extend(tags)
        data_tag_ids.extend(tag_ids)

    tq.update(1)
    tq.close()
    morph_tag_data = pd.DataFrame(
        list(zip(data_sent_idx, data_token_idx, data_tokens, data_morph_idx, data_forms, data_tags, data_tag_ids)),
        columns=['sent_idx', 'token_idx', 'token', 'morph_idx', 'form', 'tag', 'tag_id'])
    return morph_tag_data


def load_morph_vocab(data_path: Path, partition: list, include_eos: bool):
    logging.info(f'Loading morph vocab')
    char_vectors, char_vocab = load_char_vocab(data_path)
    tags, feats = set(), set()
    for part in partition:
        morph_file = data_path / f'{part}_morph.csv'
        morph_data = pd.read_csv(str(morph_file), index_col=0)
        tags |= set(morph_data.tag)
        feats |= set(morph_data.feats)
    feats.add('_')
    tags.add('_')
    special_labels = ['<pad>']
    if include_eos:
        special_labels += ['<s>', '</s>']
    feats = special_labels + sorted(list(feats))
    tags = special_labels + sorted(list(tags))
    tag2index = {t: i for i, t in enumerate(tags)}
    index2tag = {v: k for k, v in tag2index.items()}
    tag_vocab = {'tag2index': tag2index, 'index2tag': index2tag}
    feats2index = {f: i for i, f in enumerate(feats)}
    index2feats = {v: k for k, v in feats2index.items()}
    feats_vocab = {'feats2index': feats2index, 'index2feats': index2feats}
    return char_vectors, char_vocab, tag_vocab, feats_vocab


def save_morph_tag_data_samples(data_path: Path, morph_partition: dict, tag2index: dict, include_eos: bool):
    tag_samples_partition = {}
    for part in morph_partition:
        tag_samples_file = data_path / f'{part}_morph_tag_data_samples.csv'
        logging.info(f'preprocessing {part} morph tag data samples')
        samples_df = _collate_morph_tag_data_samples(morph_partition[part], tag2index, include_eos)
        logging.info(f'saving {tag_samples_file}')
        samples_df.to_csv(str(tag_samples_file))
        tag_samples_partition[part] = samples_df
    return tag_samples_partition


def _load_morph_tag_data_samples(data_path: Path, partition: list):
    tag_samples_partition = {}
    for part in partition:
        tag_samples_file = data_path / f'{part}_morph_tag_data_samples.csv'
        logging.info(f'loading {tag_samples_file}')
        samples_df = pd.read_csv(str(tag_samples_file), index_col=0)
        tag_samples_partition[part] = samples_df
    return tag_samples_partition


def load_morph_tag_data(data_path: Path, partition: list):
    morph_tag_data_samples = _load_morph_tag_data_samples(data_path, partition)
    arr_data = {}
    for part in partition:
        morph_tag_df = morph_tag_data_samples[part]
        morph_tag_data = morph_tag_df[['sent_idx', 'token_idx', 'tag_id']]
        morph_tag_data_groups = morph_tag_data.groupby('sent_idx')
        sent_arrs = []
        for sent_idx, sent_df in sorted(morph_tag_data_groups):
            morph_token_data_groups = sent_df.groupby('token_idx')
            sent_arrs.append([token_df.to_numpy() for token_id, token_df in sorted(morph_token_data_groups)])
        token_morph_size = list(set([arr.shape[0] for token_arrs in sent_arrs for arr in token_arrs]))
        token_lengths = [len(arr) for arr in sent_arrs]
        max_num_tokens = max(token_lengths)
        sent_pad_lengths = [max_num_tokens - l for l in token_lengths]
        for sent_arr, pad_len in zip(sent_arrs, sent_pad_lengths):
            sent_id = np.unique(sent_arr[-1][:, 0]).item()
            token_pad_arr = np.array([[sent_id, 0, 0]] * token_morph_size[0], dtype=np.int)
            sent_arr.extend([token_pad_arr] * pad_len)
        morph_tag_arr = np.stack(sent_arrs, axis=0)

        arr_data[part] = morph_tag_arr
    return arr_data
