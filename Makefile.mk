# Tell make to use bash and to inherit the current environment's PATH.
# This is crucial for ensuring it finds the correct conda python.
SHELL := /bin/bash
.SHELLFLAGS = -ec

# Find the python executable from the inherited PATH.
PYTHON := $(shell which python)

run:
# 	PYTHONPATH="" $(PYTHON) main.py --algo fedadam_v2 --rounds 2 --num_clients 10 --client_fraction 1.0 --local_epochs 5 --lr 0.01 --alpha 0.5 --gpu 0 --batch_size 32
	PYTHONPATH="" $(PYTHON) main.py --algo fedavg --rounds 30 --num_clients 10 --client_fraction 1 --local_epochs 3 --lr 0.01 --alpha 0.5 --gpu 0 --batch_size 32
# 	PYTHONPATH="" $(PYTHON) main.py --algo fedadam --rounds 20 --num_clients 10 --client_fraction 1.0 --local_epochs 3 --lr 0.001 --alpha 0.5 --gpu 0 --batch_size 32
# 	PYTHONPATH="" $(PYTHON) main.py --algo fedrmsprop --rounds 2 --num_clients 10 --client_fraction 1.0 --local_epochs 1 --lr 0.001 --alpha 0.5 --gpu 0 --batch_size 32
# 	PYTHONPATH="" $(PYTHON) main.py --algo fedadagrad --rounds 2 --num_clients 10 --client_fraction 1.0 --local_epochs 1 --lr 0.01 --alpha 0.5 --gpu 0 --batch_size 32