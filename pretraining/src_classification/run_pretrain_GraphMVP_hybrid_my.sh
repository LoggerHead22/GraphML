#!/usr/bin/env bash

source $HOME/.bashrc
conda activate GraphMVP

echo $@
date
echo "start"
python pretrain_GraphMVP_hybrid_my.py $@
echo "end"
date
