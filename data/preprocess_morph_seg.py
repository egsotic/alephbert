from .preprocess_base import *


def _insert_morph_ids(df: pd.DataFrame):
    sentences = sorted(df.groupby(df.sent_id))
    sent_morph_ids = [list(sent_df.reset_index(drop=True).index + 1) for sent_id, sent_df in sentences]
    morph_ids = [mid for l in sent_morph_ids for mid in l]
    df.insert(3, 'morph_id', morph_ids)
    return df


def _get_morph_df(raw_lattice_df: pd.DataFrame):
    morph_df = raw_lattice_df[['sent_id', 'token_id', 'token', 'form', 'lemma', 'tag', 'feats']]
    return _insert_morph_ids(morph_df)


def _create_morph_form_char_df(morph_df: pd.DataFrame):
    morph_form_char_df = morph_df.copy()
    morph_form_char_df = add_chars(morph_form_char_df, 'form')
    sent_groups = morph_form_char_df.groupby([morph_form_char_df.sent_id])
    num_sentences = len(sent_groups)
    morphemes = sorted(morph_form_char_df.groupby([morph_form_char_df.sent_id, morph_form_char_df.morph_id]))
    data_sent_ids, data_token_ids, data_tokens, data_morph_ids = [], [], [], []
    data_forms, data_tags, data_chars = [], [], []
    prev_sent_id, prev_token_id = 1, 1
    tq = tqdm(total=num_sentences, desc="Sentence")
    for (sent_id, morph_id), morph_df in morphemes:
        sent_ids = list(morph_df.sent_id) + [sent_id]
        token_ids = list(morph_df.token_id)
        token_ids += token_ids[:1]
        tokens = list(morph_df.token)
        tokens += tokens[:1]
        morph_ids = list(morph_df.morph_id) + [morph_id]
        forms = list(morph_df.form)
        forms += forms[:1]
        tags = list(morph_df.tag)
        tags += tags[:1]
        cur_token_id = token_ids[-1]
        if sent_id != prev_sent_id or cur_token_id != prev_token_id:
            data_chars[-1] = '</s>'
            prev_token_id = cur_token_id
            if sent_id != prev_sent_id:
                tq.update(1)
                prev_token_id = 1
                prev_sent_id = sent_id
        chars = list(morph_df.char) + ['<sep>']
        data_sent_ids.extend(sent_ids)
        data_token_ids.extend(token_ids)
        data_tokens.extend(tokens)
        data_morph_ids.extend(morph_ids)
        data_forms.extend(forms)
        data_tags.extend(tags)
        data_chars.extend(chars)
    data_chars[-1] = '</s>'
    tq.update(1)
    tq.close()
    morph_char_data = pd.DataFrame(list(zip(data_sent_ids, data_token_ids, data_tokens, data_morph_ids, data_forms,
                                            data_tags, data_chars)),
                                   columns=['sent_id', 'token_id', 'token', 'morph_id', 'form', 'tag', 'char'])
    return morph_char_data


def _collate_morph_form_char_data_samples(morph_form_char_df: pd.DataFrame, char2index: dict):
    sent_groups = morph_form_char_df.groupby([morph_form_char_df.sent_id])
    num_sentences = len(sent_groups)
    token_groups = morph_form_char_df.groupby([morph_form_char_df.sent_id, morph_form_char_df.token_id])
    max_num_chars = max([len(token_df) for _,  token_df in token_groups])
    data_sent_idx, data_token_idx, data_tokens, data_morph_idx = [], [], [], []
    data_forms, data_chars, data_char_ids = [], [], []
    cur_sent_id = None
    tq = tqdm(total=num_sentences, desc="Sentence")
    for (sent_id, token_id), token_df in sorted(token_groups):
        if cur_sent_id != sent_id:
            if cur_sent_id is not None:
                tq.update(1)
            cur_sent_id = sent_id

        sent_idxs = list(token_df.sent_id)
        token_idxs = list(token_df.token_id)
        tokens = list(token_df.token)
        morph_idxs = list(token_df.morph_id)
        forms = list(token_df.form)
        chars = list(token_df.char)
        char_ids = [char2index[c] for c in chars]

        pad_len = max_num_chars - len(chars)
        sent_idxs.extend(sent_idxs[-1:] * pad_len)
        token_idxs.extend(token_idxs[-1:] * pad_len)
        tokens.extend(tokens[-1:] * pad_len)
        morph_idxs.extend([-1] * pad_len)
        forms.extend(['<pad>'] * pad_len)
        chars.extend(['<pad>'] * pad_len)
        char_ids.extend([char2index['<pad>']] * pad_len)

        data_sent_idx.extend(sent_idxs)
        data_token_idx.extend(token_idxs)
        data_tokens.extend(tokens)
        data_morph_idx.extend(morph_idxs)
        data_forms.extend(forms)
        data_chars.extend(chars)
        data_char_ids.extend(char_ids)

    tq.update(1)
    tq.close()
    morph_char_data = pd.DataFrame(
        list(zip(data_sent_idx, data_token_idx, data_tokens, data_morph_idx, data_forms, data_chars, data_char_ids)),
        columns=['sent_idx', 'token_idx', 'token', 'morph_idx', 'form', 'char', 'char_id'])
    return morph_char_data


def get_morph_data(data_path: Path, raw_partition: dict):
    morph_partition = {}
    for part in raw_partition:
        morph_file = data_path / f'{part}_morph.csv'
        if not morph_file.exists():
            logging.info(f'preprocessing {part} morphemes')
            morph_df = _get_morph_df(raw_partition[part])
            logging.info(f'saving {morph_file}')
            morph_df.to_csv(str(morph_file))
        else:
            logging.info(f'loading {morph_file}')
            morph_df = pd.read_csv(str(morph_file), index_col=0)
        morph_partition[part] = morph_df
    return morph_partition


def get_morph_form_char_data(data_path: Path, morph_partition: dict):
    morph_form_char_partition = {}
    for part in morph_partition:
        morph_form_char_file = data_path / f'{part}_morph_form_char.csv'
        if not morph_form_char_file.exists():
            logging.info(f'preprocessing {part} morpheme form chars')
            morph_form_char_df = _create_morph_form_char_df(morph_partition[part])
            logging.info(f'saving {morph_form_char_file}')
            morph_form_char_df.to_csv(str(morph_form_char_file))
        else:
            logging.info(f'loading {morph_form_char_file}')
            morph_form_char_df = pd.read_csv(str(morph_form_char_file), index_col=0)
        morph_form_char_partition[part] = morph_form_char_df
    return morph_form_char_partition


def save_morph_form_char_data_samples(data_path: Path, morph_form_char_partition: dict, char2index: dict):
    form_char_samples_partition = {}
    for part in morph_form_char_partition:
        form_char_samples_file = data_path / f'{part}_morph_form_char_data_samples.csv'
        logging.info(f'preprocessing {part} morph form char data samples')
        samples_df = _collate_morph_form_char_data_samples(morph_form_char_partition[part], char2index)
        logging.info(f'saving {form_char_samples_file}')
        samples_df.to_csv(str(form_char_samples_file))
        form_char_samples_partition[part] = samples_df
    return form_char_samples_partition


def _load_morph_form_char_data_samples(data_path: Path, partition: list):
    form_char_samples_partition = {}
    for part in partition:
        form_char_samples_file = data_path / f'{part}_morph_form_char_data_samples.csv'
        logging.info(f'loading {form_char_samples_file}')
        samples_df = pd.read_csv(str(form_char_samples_file), index_col=0)
        form_char_samples_partition[part] = samples_df
    return form_char_samples_partition


def load_morph_seg_data(data_path: Path, partition: list):
    form_char_data_samples = _load_morph_form_char_data_samples(data_path, partition)
    return to_sub_token_seq(form_char_data_samples, 'char_id')