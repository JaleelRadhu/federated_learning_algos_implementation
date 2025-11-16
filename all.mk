# This Makefile runs a hyperparameter sweep for all implemented federated learning algorithms.

# --- Configuration ---
# Define the algorithms to test. These must match the keys in the ALGORITHMS dict in main.py
# ALGORITHMS := fedavg fedadam_v2 fedrmsprop fedadagrad fedamsgrad fedadamw

ALGORITHMS :=   fedavg
# Note: 'fedadam' (client-side adam) is excluded as it is known to perform poorly and serves as a baseline comparison.
# You can add it back to the list if you wish to generate its results.

# Define the hyperparameter space to explore

ROUNDS := 30
NUM_CLIENTS := 10 20 30
BATCH_SIZES := 32 16 128
CLIENT_FRACTIONS :=1 0.8 0.5
ALPHAS := 0.5 1
LRS    :=  0.001 0.01
EPOCHS := 1 2 3

# --- Environment Setup ---
# Tell make to use bash and to inherit the current environment's PATH.
SHELL := /bin/bash
.SHELLFLAGS = -ec

# Find the python executable from the inherited PATH.
PYTHON ?= $(shell which python)

# --- Main Target ---
# Phony target to avoid conflicts with a file named 'run-all'
.PHONY: run-all
#include batch size in outermost loop
run-all:
	@echo "--- Starting Hyperparameter Sweep ---"
	@for rounds in $(ROUNDS); do \
		for num_clients in $(NUM_CLIENTS); do \
			for batch_size in $(BATCH_SIZES); do \
				for client_fraction in $(CLIENT_FRACTIONS); do \
					for alpha in $(ALPHAS); do \
						for lr in $(LRS); do \
							for epochs in $(EPOCHS); do \
								echo -e "\n\n--- Running Hyperparameter Set: R=$$rounds, C=$$num_clients, F=$$client_fraction, BS=$$batch_size, A=$$alpha, LR=$$lr, E=$$epochs ---"; \
								for algo in $(ALGORITHMS); do \
									echo -e ">>> Algorithm: $$algo"; \
									PYTHONPATH="" $(PYTHON) main.py --algo $$algo --rounds $$rounds --local_epochs $$epochs --lr $$lr --alpha $$alpha --num_clients $$num_clients --client_fraction $$client_fraction --gpu 1 --batch_size $$batch_size; \
								done; \
							done; \
						done; \
					done; \
				done; \
			done; \
		done; \
	done
	@echo -e "\n--- Hyperparameter Sweep Finished ---"
