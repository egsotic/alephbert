import logging
from pathlib import Path
import pandas as pd

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def read_file(path):
    with open(str(path)) as f:
        return [line.strip() for line in f.readlines()]


def parse_line(line: str) -> dict:
    parts = line.split(':')
    info_parts = parts[0].split()
    score_parts = line.split()[-6:]
    data_partition_name = info_parts[0][1:]
    eval_type = info_parts[3]
    morph_type = ''.join([info_parts[i] for i in range(4, len(info_parts)-2)])
    morph_type = [t[1:-1] for t in tuple(map(str, morph_type.split(',')))]
    morph_type[0] = morph_type[0][1:]
    morph_type[-1] = morph_type[-1][:-1]
    if len(morph_type[-1]) == 0:
        morph_type = morph_type[:-1]
    precision = float(score_parts[-5][:-1])
    recall = float(score_parts[-3][:-1])
    f1 = float(score_parts[-1][:-5])
    return {'partition': data_partition_name, 'eval_type': eval_type, 'morph_type': tuple(morph_type),
            'precision': precision, 'recall': recall, 'f1': f1}


def parse_input_file(input_file_path: Path) -> pd.DataFrame:
    lines = read_file(input_file_path)
    parsed_eval_scores = [parse_line(line) for line in lines if 'eval scores' in line]
    return pd.DataFrame(parsed_eval_scores)


def insert_eval_info(df, crf, task, tb, epochs, vocab, corpus, tokenizer, model_size, model_name):
    num_morph_types = len(df['morph_type'].unique())
    df.insert(0, 'iter', [int(i // (4 * num_morph_types)) + 1 for i in range(len(df))])
    df.insert(0, 'crf', crf)
    df.insert(0, 'task', task)
    df.insert(0, 'tb', tb.upper())
    df.insert(0, 'epochs', epochs)
    df.insert(0, 'vocab', vocab)
    df.insert(0, 'corpus', corpus)
    df.insert(0, 'tokenizer', tokenizer)
    df.insert(0, 'model_size', model_size)
    df.insert(0, 'model_name', model_name)


def fix_partition(df):
    num_morph_types = len(df['morph_type'].unique())
    rows = [j for i, j in enumerate(list(range(0, len(df), 2 * num_morph_types))) if (i % 2) == 1]
    for j in range(2 * num_morph_types):
        for i in [row + j for row in rows]:
            df.at[i, 'partition'] = 'test'


def get_tb_eval_dataframes(bert_root_path, bert_version, task, epochs, vocab, corpus, tokenizer, model_size,
                           model_name) -> list:
    tb_dataframes = []
    for schema in ['spmrl', 'ud']:
        if schema == 'spmrl':
            tb = 'hebtb'
            if 'ner' in morph_task:
                schema_path = 'for_amit_spmrl'
            else:
                schema_path= 'HebrewTreebank'
        elif 'ner' in morph_task:
            continue
        else:
            tb = 'HTB'
            schema_path = 'UD_Hebrew'
        tb_path = bert_root_path / schema_path / tb
        nb_file_name = f'{bert_version}-{schema}-{task.replace("_", "-")}.ipynb'
        nb_file_path = tb_path / nb_file_name
        eval_scores_df = parse_input_file(nb_file_path)
        insert_eval_info(eval_scores_df, False, task, schema, epochs, vocab, corpus, tokenizer, model_size, model_name)
        tb_dataframes.append(eval_scores_df)
        if model_size == 'small' and corpus_name == 'oscar' and vocab_size == 32000:
            fix_partition(eval_scores_df)
            continue
        if 'ner' in task:
            nb_crf_file_name = f'{bert_version}-{schema}-{task.replace("_", "-")}-crf.ipynb'
            nb_crf_file_path = tb_path / nb_crf_file_name
            eval_scores_df = parse_input_file(nb_crf_file_path)
            insert_eval_info(eval_scores_df, True, task, schema, epochs, vocab, corpus, tokenizer, model_size, model_name)
            tb_dataframes.append(eval_scores_df)
    return tb_dataframes


dataframes = []
morph_path = Path('experiments') / 'morph'
for model_name in ['alephbert', 'hebert', 'mbert']:
    for morph_task in ['seg_only', 'seg_tag', 'seg_tag_feats', 'seg_ner', 'seg_tag_ner']:
        task_path = morph_path / f'morph_{morph_task}' / 'bert'
        for model_size in ['small', 'basic']:
            model_size_path = task_path / model_size
            for tokenizer_type in ['wordpiece']:
                tokenizer_type_path = model_size_path / tokenizer_type
                if model_name in ['hebert', 'mbert']:
                    bert_version = model_name
                    bert_version_path = tokenizer_type_path / bert_version
                    if not bert_version_path.exists():
                        continue
                    dataframes.extend(get_tb_eval_dataframes(bert_version_path, bert_version, morph_task, 0, 0, 'None',
                                                             tokenizer_type, model_size, model_name))
                else:
                    for corpus_name in ['oscar', 'owt']:
                        for vocab_size in [32000, 52000, 104000]:
                            for epochs in ['05', '10', '15']:
                                bert_version = f'bert-{model_size}-{tokenizer_type}-{corpus_name}-{vocab_size}-{epochs}'
                                bert_version_path = tokenizer_type_path / bert_version
                                if not bert_version_path.exists():
                                    continue
                                dataframes.extend(get_tb_eval_dataframes(bert_version_path, bert_version, morph_task,
                                                                         epochs, vocab_size, corpus_name,
                                                                         tokenizer_type, model_size, model_name))
pd.concat(dataframes).to_csv('morph-eval.csv', index=False)
