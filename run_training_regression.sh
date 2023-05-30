#!/bin/bash

for dataset in FreeSolv ESOL Lipophilicity QM7 QM8 QM9
do 
	if [ "${dataset:0:2}" == "QM" ]
	then 
		metric="mae";
	else
		metric="rmse";
	fi
	
	python gnn/train.py --data_path dataset/$dataset/smiles_w_targets.csv --split_type scaffold_balanced --seed 22 --num_folds 5 --dataset_type regression \
	--save_dir checkpoints/${dataset}_5f_sf_rdkit --metric $metric --extra_metric r2 --features_generator rdkit_2d_normalized --no_features_scaling --split_sizes 0.8 0.05 0.15

	python gnn/train.py --data_path dataset/$dataset/smiles_w_targets.csv  --seed 22 --num_folds 5 --dataset_type regression \
        --save_dir checkpoints/${dataset}_5f_rdkit --metric $metric --extra_metric r2 --features_generator rdkit_2d_normalized --no_features_scaling --split_sizes 0.8 0.05 0.15

	python gnn/train.py --data_path dataset/$dataset/smiles_w_targets.csv --split_type scaffold_balanced --seed 22 --num_folds 5 --dataset_type regression \
        --save_dir checkpoints/${dataset}_5f_sf_our --metric $metric --extra_metric r2 --features_path dataset/${dataset}/matrix_alphabet_NNdb_2chains.csv --split_sizes 0.8 0.05 0.15
	
	python gnn/train.py --data_path dataset/$dataset/smiles_w_targets.csv --seed 22 --num_folds 5 --dataset_type regression \
        --save_dir checkpoints/${dataset}_5f_our --metric $metric --extra_metric r2 --features_path dataset/${dataset}/matrix_alphabet_NNdb_2chains.csv --split_sizes 0.8 0.05 0.15

	python gnn/train.py --data_path dataset/$dataset/smiles_w_targets.csv --split_type scaffold_balanced --seed 22 --num_folds 5 --dataset_type regression \
        --save_dir checkpoints/${dataset}_5f_sf --metric $metric --extra_metric r2  --split_sizes 0.8 0.05 0.15
	
	python gnn/train.py --data_path dataset/$dataset/smiles_w_targets.csv  --seed 22 --num_folds 5 --dataset_type regression \
        --save_dir checkpoints/${dataset}_5f --metric $metric --extra_metric r2  --split_sizes 0.8 0.05 0.15
done
	
