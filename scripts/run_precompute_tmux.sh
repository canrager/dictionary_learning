#!/bin/bash

# Script to run precompute_activations.py in a detached tmux session

SESSION_NAME="precompute_activations"
SCRIPT_PATH="/home/can/dictionary_learning/training_demo/precompute_activations.py"
VENV_PATH="/home/can/dictionary_learning/.venv/bin/python"

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists and is running in the background."
    echo "To check status: tmux capture-pane -t $SESSION_NAME -p"
    echo "To attach: tmux attach -t $SESSION_NAME"
    echo "To kill: tmux kill-session -t $SESSION_NAME"
else
    echo "Creating new detached tmux session '$SESSION_NAME'..."
    # Create new detached tmux session and run the script
    tmux new-session -d -s "$SESSION_NAME" -c "/home/can/dictionary_learning" \
        "$VENV_PATH $SCRIPT_PATH"
    
    echo "Session '$SESSION_NAME' started in background."
    echo "Commands:"
    echo "  Check output: tmux capture-pane -t $SESSION_NAME -p"
    echo "  Attach: tmux attach -t $SESSION_NAME"
    echo "  Kill: tmux kill-session -t $SESSION_NAME"
fi