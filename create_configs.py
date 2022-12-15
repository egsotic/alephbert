import argparse
import itertools
import json
import logging
import os
from pathlib import Path

import tqdm


def load_config(path: Path):
    with open(path) as f:
        return json.load(f)


def main(config):
    # config
    template_config_path = config['template_config_path']
    placeholder_config_path = config['placeholder_config_path']

    out_dir_path = Path(config['out_dir_path'])
    os.makedirs(out_dir_path, exist_ok=True)

    override_existing = config['override_existing']

    # load
    template_config = load_config(template_config_path)
    placeholder_config = load_config(placeholder_config_path)

    name_template = template_config['name']
    template = template_config['template']
    placeholder_params = placeholder_config

    for values in tqdm.tqdm(itertools.product(*placeholder_params.values())):
        # placeholders
        placeholders = {}
        for k, v in zip(placeholder_params.keys(), values):
            if isinstance(v, list):
                for i, _v in enumerate(v):
                    placeholders[f'{k}__{i}'] = _v
            else:
                placeholders[k] = v

        # name
        name = name_template.format(**placeholders)
        conf_path = out_dir_path / name

        # skip
        if conf_path.exists() and not override_existing:
            print('skipping existing', conf_path)
            continue

        # conf
        conf = {}

        for k, v in template.items():
            # special rule
            if '@' in k:
                key, func = k.split('@')
                # keep as value, no string formatting (e.g. list)
                if func == 'value':
                    conf[key] = placeholders[v]
                else:
                    print(f'unknown func {func} in key {k}')
            # regular case
            else:
                conf[k] = v

                # string formatting
                if isinstance(v, str):
                    conf[k] = conf[k].format(**placeholders)

        # save
        with open(conf_path, 'w') as f:
            json.dump(conf, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=Path, help='config path')
    args = parser.parse_args()

    config_path = args.config_path

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    logging.info(config)

    main(config)
