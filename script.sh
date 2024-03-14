#!/bin/bash

#PBS -lwalltime=20:00:00
#PBS -lselect=1:ncpus=N:mem=Mgb

git clone https://github.com/Parvfect/Cluster.git
cd Cluster
python3 -m venv env
source env/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
python coupon_collector.py
cp -r $TMPDIR/Cluster/Run* $HOME/code_run/