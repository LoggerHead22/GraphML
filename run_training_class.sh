#!/bin/bash

metric="$1"

echo $metric

for dataset in HIV BACE BBBP Tox21 SIDER ClinTox
do 	
	python gnn/train.py --data_path dataset/$dataset/smiles_w_targets.csv --split_type scaffold_balanced --seed 22 --num_folds 5 --dataset_type classification \
	--save_dir checkpoints_class/${dataset}_5f_sf_rdkit --metric $metric  --features_generator rdkit_2d_normalized --no_features_scaling --split_sizes 0.8 0.05 0.15

	python gnn/train.py --data_path dataset/$dataset/smiles_w_targets.csv  --seed 22 --num_folds 5 --dataset_type classification \
        --save_dir checkpoints_class/${dataset}_5f_rdkit --metric $metric  --features_generator rdkit_2d_normalized --no_features_scaling --split_sizes 0.8 0.05 0.15

	python gnn/train.py --data_path dataset/$dataset/smiles_w_targets.csv --split_type scaffold_balanced --seed 22 --num_folds 5 --dataset_type classification \
        --save_dir checkpoints_class/${dataset}_5f_sf_our --metric $metric --features_path dataset/${dataset}/matrix_alphabet_NNdb_2chains.csv --split_sizes 0.8 0.05 0.15
	
	python gnn/train.py --data_path dataset/$dataset/smiles_w_targets.csv --seed 22 --num_folds 5 --dataset_type classification \
        --save_dir checkpoints_class/${dataset}_5f_our --metric $metric --features_path dataset/${dataset}/matrix_alphabet_NNdb_2chains.csv --split_sizes 0.8 0.05 0.15

	python gnn/train.py --data_path dataset/$dataset/smiles_w_targets.csv --split_type scaffold_balanced --seed 22 --num_folds 5 --dataset_type classification \
        --save_dir checkpoints_class/${dataset}_5f_sf --metric $metric --split_sizes 0.8 0.05 0.15
	
	python gnn/train.py --data_path dataset/$dataset/smiles_w_targets.csv  --seed 22 --num_folds 5 --dataset_type classification \
        --save_dir checkpoints_class/${dataset}_5f --metric $metric  --split_sizes 0.8 0.05 0.15
done
	
