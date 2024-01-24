#!/bin/bash

python -m venv random_env
.\random_env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python coupon_collector.py
exit