# Using Transformer-based Models for Automated ICD Coding: Extreme Multi-label Long Text Classification

The source code is for the research: [Automated ICD Coding using Extreme Multi-label Long Text Transformer-based Models](https://arxiv.org/abs/2212.05857).

## Prerequisites
Restore [MIMIC-III v1.4 data](https://physionet.org/content/mimiciii/1.4/) into a Postgres database. 

## 1. Further pretraining BIGBIRD
### Prepare training data
python3 lm.prepare_mimic_data_for_pretrain_lm.py

### Pretraining
python3 lm/run_mlm.py \
    --model_name_or_path=google/bigbird-roberta-base \
    --train_file=[data_dir]/mimic3_uncased_preprocessed_total.txt \
    --output_dir=[model_dir] \
    --max_seq_length=4096 \
    --line_by_line=True \
    --do_train=True \
    --do_eval=True \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=32 \
    --overwrite_output_dir=True \
    --max_steps=500000 \
    --save_steps=1000 \
    --warmup_steps=10000 \
    --learning_rate=2e-05 \
    --save_total_limit=20
### ClinicalBIGBIRD
The pretrained ClinicalBIGBIRD can be downloaded [here](https://unsw-my.sharepoint.com/:f:/g/personal/z5250377_ad_unsw_edu_au/EioZr7d0at9Krjlm37mz-NkBUg6l__57LREL-XK-fFOAjw?e=yAkWrL).

## 2. ICD coding - Baseline model
### Data preparation
use mimic3_data_preparer.py in [HiLAT](https://github.com/leiboliu/HiLAT/tree/main/hilat/data) to prepare the raw training data with the following command line flags:

|         Name       |  Value |
| ------------------ | ------ |
| pre_process_level  | raw    |
| segment_text       | False  |

### Train model
1. Download the pretrained Transformer [RoBERTa-PM-M3](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-distill-align-hf.tar.gz)
2. Training
python3 xr-lat/run_coding.py config-baseline.json

## 3. ICD coding - XR-LAT
Data preparation is the same as the baseline model.

### Generate hierarchical code tree (HCT) on MIMIC-III-full dataset
python3 data_processing/generate_hierarchical_label_tree.py

### Train model
python3 xr-lat/run_coding.py config-xrlat.json

### 4. ICD coding - XR-Transformer
1. Prepare training data
python3 data_processing.prepare_xrtransformer_data.py
2. Install [libpecos](https://github.com/amzn/pecos)
Use install and develop locally option.
3. Replace /pecos/xmc/xtransformer/network.py by xr-transformer.network.py
4. Create TF-IDF features

python3 -m pecos.utils.featurization.text.preprocess build --text-pos 0 --input-text-path ../data/mimic3/full-xr/train.txt --output-model-folder ./tfidf-model

python3 -m pecos.utils.featurization.text.preprocess run --text-pos 0 --input-preprocessor-folder ./tfidf-model --input-text-path ../data/mimic3/full-xr/train.txt --output-inst-path ./data/train.tfidf.npz

python3 -m pecos.utils.featurization.text.preprocess run --text-pos 0 --input-preprocessor-folder ./tfidf-model --input-text-path ../data/mimic3/full-xr/test.txt --output-inst-path ./data/test.tfidf.npz

5. Training

xr-transformer/train_and_predict.sh

