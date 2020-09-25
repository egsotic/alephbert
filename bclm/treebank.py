# import sys
# sys.path.insert(0, "/Users/Amit/dev/aseker00/AlephBert/src/bclm")
from pathlib import Path
import pandas as pd
import logging

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


def main():
    gold_partition = spmrl('data/raw', '/Users/Amit/dev/onlplab/HebrewResources')
    gold_partition = spmrl('data/raw')
    # ma_partition = spmrl('data/raw', '/Users/Amit/dev/onlplab/HebrewResources', ma_name='heblex')


if __name__ == '__main__':
    main()
