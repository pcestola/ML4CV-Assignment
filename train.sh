#!/bin/bash

# Controlla se i parametri sono stati forniti
if [ "$#" -lt 2 ]; then
    echo "Uso: $0 <nome_sessione_tmux> <comando_python>"
    echo "Esempio: $0 train_5 'python train.py'"
    exit 1
fi

# Nome della sessione tmux e comando Python
SESSION_NAME=$1
PYTHON_COMMAND=$2

# Nome dell'ambiente Conda (sostituisci con il tuo ambiente)
CONDA_ENV="ml4cv"

# Avvia una nuova sessione tmux in background
tmux new-session -d -s $SESSION_NAME

# Esegui il comando conda e poi il comando Python nella sessione tmux
tmux send-keys -t $SESSION_NAME "conda activate $CONDA_ENV" C-m
tmux send-keys -t $SESSION_NAME "$PYTHON_COMMAND" C-m