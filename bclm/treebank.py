# import sys
# sys.path.insert(0, "/Users/Amit/dev/aseker00/AlephBert/src/bclm")
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import logging
import itertools
from bclm.format import lattice


def spmrl(data_root_path, tb_root_path=None, tb_name='hebtb', ma_name=None):
    partition = {'train': None, 'dev': None, 'test': None}
    ma_type = ma_name if ma_name is not None else 'gold'
    data_tb_path = Path(data_root_path) / tb_name / ma_type
    # if not local_tb_path.exists():
    if tb_root_path is not None:
        data_tb_path.mkdir(parents=True, exist_ok=True)
        logging.info(f'loading treebank: {tb_root_path}')
        partition = lattice.load_spmrl(Path(tb_root_path), partition, tb_name, ma_name)
        for part in partition:
            lattice_file_path = data_tb_path / f'{part}_{tb_name}-{ma_type}.lattices.csv'
            logging.info(f'saving: {lattice_file_path}')
            partition[part].to_csv(lattice_file_path)
    else:
        for part in partition:
            lattice_file_path = data_tb_path / f'{part}_{tb_name}-{ma_type}.lattices.csv'
            logging.info(f'loading: {lattice_file_path}')
            partition[part] = pd.read_csv(lattice_file_path, index_col=0)
        # partition = {part: pd.read_csv(data_root_path / f'{part}_{tb_name}-{ma_type}.lattices.csv', index_col=0) for part in partition}
    # tb = {}
    # for part in partition:
    #     df = partition[part]
    #     gb = df.groupby(df.sent_id)
    #     tb[part] = [gb.get_group(x).reset_index(drop=True) for x in gb.groups]
    #     logging.info(f'{tb_name} {part} lattices: {len(tb[part])}')
    # return tb
    return partition


def get_subsets(s, n):
    return list(itertools.combinations(s, n))


def morph_eval(gold_df, pred_df, fields):
    gold_gb = gold_df.groupby([gold_df.sent_id, gold_df.token_id])
    pred_gb = pred_df.groupby([pred_df.sent_id, pred_df.token_id])
    aligned_gold_counts, aligned_pred_counts, aligned_intersection_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    mset_gold_counts, mset_pred_counts, mset_intersection_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    for (sent_id, token_id), gold in sorted(gold_gb):
        for n in range(1, len(fields) + 1):
            fsets = get_subsets(fields, n)
            for fs in fsets:
                gold_values = [tuple(row[1].values) for row in gold[list(fs)].iterrows()]
                if (sent_id, token_id) not in pred_gb.groups:
                    pred_values = []
                else:
                    pred = pred_gb.get_group((sent_id, token_id))
                    pred_values = [tuple(row[1].values) for row in pred[list(fs)].iterrows()]
                # mset
                gold_count, pred_count = Counter(gold_values), Counter(pred_values)
                intersection_count = gold_count & pred_count
                mset_gold_counts[fs] += sum(gold_count.values())
                mset_pred_counts[fs] += sum(pred_count.values())
                mset_intersection_counts[fs] += sum(intersection_count.values())
                # aligned
                intersection_values = [p for g, p in zip(gold_values, pred_values) if p == g]
                aligned_gold_counts[fs] += len(gold_values)
                aligned_pred_counts[fs] += len(pred_values)
                aligned_intersection_counts[fs] += len(intersection_values)
    aligned_scores, mset_scores = {}, {}
    for fs in aligned_gold_counts:
        precision = aligned_intersection_counts[fs] / aligned_pred_counts[fs] if aligned_pred_counts[fs] else 0.0
        recall = aligned_intersection_counts[fs] / aligned_gold_counts[fs] if aligned_gold_counts[fs] else 0.0
        f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0.0
        aligned_scores[fs] = precision, recall, f1
    for fs in mset_gold_counts:
        precision = mset_intersection_counts[fs] / mset_pred_counts[fs] if mset_pred_counts[fs] else 0.0
        recall = mset_intersection_counts[fs] / mset_gold_counts[fs] if mset_gold_counts[fs] else 0.0
        f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0.0
        mset_scores[fs] = precision, recall, f1
    return aligned_scores, mset_scores


def main():
    gold_partition = spmrl('data/raw', '/Users/Amit/dev/onlplab/HebrewResources')
    gold_partition = spmrl('data/raw')
    # ma_partition = spmrl('data/raw', '/Users/Amit/dev/onlplab/HebrewResources', ma_name='heblex')


if __name__ == '__main__':
    main()
