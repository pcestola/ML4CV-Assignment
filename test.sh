#!/bin/bash

# Controlla se ci sono esattamente 5 argomenti (1 nome sessione + 4 comandi)
if [ "$#" -ne 5 ]; then
    echo "Uso: $0 <nome_sessione_tmux> <comando1> <comando2> <comando3> <comando4>"
    echo "Esempio: $0 test_0 'python script1.py' 'python script2.py' 'python script3.py' 'python script4.py'"
    exit 1
fi

# Nome della sessione tmux
SESSION_NAME=$1

# Comandi da eseguire
CMD1=$2
CMD2=$3
CMD3=$4
CMD4=$5

# Nome dell'ambiente Conda (sostituisci con il tuo ambiente)
CONDA_ENV="ml4cv"

# Crea una nuova sessione tmux in background
tmux new-session -d -s "$SESSION_NAME"

# Esegui il comando conda e poi il comando Python nella sessione tmux
tmux send-keys -t $SESSION_NAME "conda activate $CONDA_ENV" C-m

# Esegui il primo comando e attendi il completamento
tmux send-keys -t "$SESSION_NAME" "$CMD1" C-m

# Esegui il secondo comando e attendi il completamento
tmux send-keys -t "$SESSION_NAME" "$CMD2" C-m

# Esegui il terzo comando e attendi il completamento
tmux send-keys -t "$SESSION_NAME" "$CMD3" C-m

# Esegui il quarto comando e attendi il completamento
tmux send-keys -t "$SESSION_NAME" "$CMD4" C-m

echo "Comandi inviati alla sessione tmux '$SESSION_NAME'."
echo "Puoi collegarti con: tmux attach -t $SESSION_NAME"
