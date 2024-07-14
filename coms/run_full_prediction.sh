#!/bin/bash

python run_prediction_full.py --path1 "simple_phobert-base_4.pth" --source_len_1 256 --path2 "simple_phobert-base_19.pth" --source_len_2 256 --path3 "bartpho-syllable-base_2.pth"
python run_prediction_full.py --path1 "simple_phobert-base_4.pth" --source_len_1 256 --path2 "simple_visobert_7.pth" --source_len_2 512 --path3 "bartpho-syllable-base_2.pth" 
python run_prediction_full.py --path1 "simple_visobert_12.pth" --source_len_1 512 --path2 "simple_phobert-base_19.pth" --source_len_2 256 --path3 "bartpho-syllable-base_2.pth"

python run_prediction_full.py --path1 "simple_phobert-base_4.pth" --source_len_1 256 --path2 "simple_phobert-base_19.pth" --source_len_2 256 --path3 "bartpho-word-base_2.pth"
python run_prediction_full.py --path1 "simple_phobert-base_4.pth" --source_len_1 256 --path2 "simple_visobert_7.pth" --source_len_2 512 --path3 "bartpho-word-base_2.pth" 
python run_prediction_full.py --path1 "simple_visobert_12.pth" --source_len_1 512 --path2 "simple_phobert-base_19.pth" --source_len_2 256 --path3 "bartpho-word-base_2.pth"

python run_prediction_full.py --path1 "simple_phobert-base_4.pth" --source_len_1 256 --path2 "simple_phobert-base_19.pth" --source_len_2 256 --path3 "vit5-base_2.pth"
python run_prediction_full.py --path1 "simple_phobert-base_4.pth" --source_len_1 256 --path2 "simple_visobert_7.pth" --source_len_2 512 --path3 "vit5-base_2.pth" 
python run_prediction_full.py --path1 "simple_visobert_12.pth" --source_len_1 512 --path2 "simple_phobert-base_19.pth" --source_len_2 256 --path3 "vit5-base_2.pth"


# task 1, task 2
# phobert, visobert
python run_prediction_full.py --path1 "simple_distilbert-base-multilingual-cased_11.pth" --source_len_1 512 --path2 "simple_CafeBERT_8.pth" --source_len_2 512 --path3 "None"
python run_prediction_full.py --path1 "simple_distilbert-base-multilingual-cased_11.pth" --source_len_1 512 --path2 "simple_bert-base-multilingual-cased_1.pth" --source_len_2 512 --path3 "None"
python run_prediction_full.py --path1 "simple_distilbert-base-multilingual-cased_11.pth" --source_len_1 512 --path2 "simple_xlm-roberta-base_1.pth" --source_len_2 512 --path3 "None"

python run_prediction_full.py --path1 "simple_CafeBERT_9.pth" --source_len_1 512 --path2 "simple_distilbert-base-multilingual-cased_3.pth" --source_len_2 512 --path3 "None"
python run_prediction_full.py --path1 "simple_bert-base-multilingual-cased_8.pth" --source_len_1 512 --path2 "simple_distilbert-base-multilingual-cased_3.pth" --source_len_2 512 --path3 "None"
python run_prediction_full.py --path1 "simple_xlm-roberta-base_3.pth" --source_len_1 512 --path2 "simple_distilbert-base-multilingual-cased_3.pth" --source_len_2 512 --path3 "None"