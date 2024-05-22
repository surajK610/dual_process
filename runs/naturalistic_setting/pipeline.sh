#!/bin/bash
DATE=$(date +%m-%d)
SEED=seed_3

export LEARNING_DYNAMICS_HOME=/home
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments

# Activate Python environment
source $LEARNING_DYNAMICS_HOME/venv/bin/activate 

# Define the steps as an array
steps=(0 20 40 60 80 100 120 140 160 180 200 300 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000)

# Loop through all the steps
for step_index in ${!steps[@]}
do
    step=${steps[$step_index]}

    # Setup output directories and log files
    output_dir="outputs/icl_bert/${SEED}/"
    log_dir="${output_dir}slurm_out/"
    mkdir -p $log_dir

    log_file="${log_dir}log_${step_index}.out"
    err_file="${log_dir}log_${step_index}.err"

    echo "Running ICL Experiment with step: $step"

    # Execute the Python script and redirect output and errors to log files
    python3 $EXPERIMENT_SRC_DIR/icl_bert.py --epochs 4 --batch_size 32 --num_examples_test 10000 --model_name "google/multiberts-${SEED}-step_${step}k" --output_dir "$output_dir" > "$log_file" 2> "$err_file"
done
