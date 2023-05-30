# Pre-training GNN for Molecular Property Prediction

## Настройка окружения

Установите окрежение через conda env

```bash
conda create -n GraphMVP python=3.7
conda activate GraphMVP

conda install -y -c rdkit rdkit
conda install -y -c pytorch pytorch=1.9.1
conda install -y numpy networkx scikit-learn
pip install ase
pip install git+https://github.com/bp-kelley/descriptastorus
pip install ogb
export TORCH=1.9.0
export CUDA=cu102  # cu102, cu110

wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
pip install torch-geometric==1.7.2
```

## Обработка данных

For dataset download, please follow the instruction [here](https://github.com/chao1224/GraphMVP/tree/main/datasets).

For data preprocessing (GEOM), please use the following commands:
```
cd src_classification
python GEOM_dataset_preparation.py --n_mol 50000 --n_conf 5 --n_upper 1000 --data_folder $SLURM_TMPDIR
cd ..

cd src_regression
python GEOM_dataset_preparation.py --n_mol 50000 --n_conf 5 --n_upper 1000 --data_folder $SLURM_TMPDIR
cd ..

mv $SLURM_TMPDIR/GEOM datasets
```
