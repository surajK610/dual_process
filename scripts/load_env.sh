#!/bin/bash

# interact -n 20 -t 01:00:00 -m 10g -p 3090-gcondo
export LEARNING_DYNAMICS_HOME=/home

# module load python cuda
source $LEARNING_DYNAMICS_HOME/venv/bin/activate 
alias activate="source $LEARNING_DYNAMICS_HOME/venv/bin/activate"