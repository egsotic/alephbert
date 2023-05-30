from typing import Union, Type

MODULE_PATH = '/home/nlp/egsotic/repo/academic-budget-bert'

import sys
sys.path.append(MODULE_PATH)


def get_model(bert_model_path: str, model_cls: Union[Type, str] = None, **kwargs):
    if type(model_cls) != str:
        model_cls = model_cls.__name__
        model_cls = model_cls.split('For')[-1]

    if model_cls == 'BertModelWrapper':
        from pretraining.modeling import BertModelWrapper
        return BertModelWrapper.from_pretrained(bert_model_path, **kwargs)
    if model_cls == 'TokenClassification':
        from pretraining.modeling import BertForTokenClassification
        return BertForTokenClassification.from_pretrained(bert_model_path, **kwargs)
    elif model_cls == 'SequenceClassification':
        from pretraining.modeling import BertForSequenceClassification
        return BertForSequenceClassification.from_pretrained(bert_model_path, **kwargs)

    raise Exception(f'unknown model_cls {model_cls}')
