nohup env CUDA_VISIBLE_DEVICES=0 python morph_train.py --config_path configs/generated/UD-v2.11-ner/train_AlephBERT_Hebrew_HTB_nemo.v_00.json >> logs/train_AlephBERT_Hebrew_HTB_nemo.v_00.log &
nohup env CUDA_VISIBLE_DEVICES=1 python morph_train.py --config_path configs/generated/UD-v2.11-ner/train_AlephBERTGimel_Hebrew_HTB_nemo.v_00.json >> logs/train_AlephBERTGimel_Hebrew_HTB_nemo.v_00.log &
nohup env CUDA_VISIBLE_DEVICES=2 python morph_train.py --config_path configs/generated/UD-v2.11-ner/train_heBERT_Hebrew_HTB_nemo.v_00.json >> logs/train_heBERT_Hebrew_HTB_nemo.v_00.log &
nohup env CUDA_VISIBLE_DEVICES=3 python morph_train.py --config_path configs/generated/UD-v2.11-ner/train_mBERT_Hebrew_HTB_nemo.v_00.json >> logs/train_mBERT_Hebrew_HTB_nemo.v_00.log &

nohup env CUDA_VISIBLE_DEVICES=3 python morph_train.py --config_path configs/generated/UD-v2.11-ner/train_XLM_Hebrew_HTB_nemo.v_00.json >> logs/train_XLM_Hebrew_HTB_nemo.v_00.log &