# Dual Process Learning: Controlling use of In-Context vs. In-Weights Strategies with Weight Forgetting

To generate the virtual environment, first run `./scripts/env_setup.sh` to generate the virtual environment. Then, run `source ./scripts/load_env.sh` loads necessary modules and the virtual environment. All experiments were conducted on RTX A5000s, each with 24 GB of GPU memory with a SLURM environment. 

For all scripts, please modify the variable `LEARNING_DYNAMICS_HOME` in the script you are trying to use. All please ensure that the output paths, log paths, and data paths exist. The `start_jupter.sh` script helps start a jupyter notebook with a remote SLURM environment by piping it to a local http server. 


## Naturalistic Experiments (Section 2)

We use the Penn Tree Bank-3 (Marcus et al. 1993) to train our probes and generate our evaluation dataset. You may find the dataset [here](https://catalog.ldc.upenn.edu/LDC99T42). Once you have the `.conllx` files, the `load_conll_dataset` function from `src/utils/data_utils.py` can help extract useful fields (this is done for you if you just run the experiments).

In order to run the naturalistic experiments, please use scripts located in `runs/naturalistic_setting`. To modify the parameters for the experiments, please modify `slurm_pipeline.sh`. For this set of experiments, we also have a `pipeline.sh` for non-SLURM enviornments

## Synthetic Experiments (Section 3)
Our synthetic experiments use the dataset generation classes are located in the `src/utils/toy_utils.py` based on specified distributional parameters. 
To run the synthetic experiments, please use the `slurm_pipeline.sh` script located in `runs/synthetic_setting`. Additional parameters are located in `src/experiments/toy_model`.

## Active Forgetting (Section 4)

To run the active forgetting experiments (Chen et al. 2024), please use the `slurm_pipeline_forgetting.sh` script located in `runs/synthetic_setting`. Additional parameters are located in `src/experiments/toy_model.py`.

## Temporary Forgetting (Section 5)

To run the temporary forgetting experiments, please use the `slurm_pipeline_temporary_forgetting.sh` script located in `runs/synthetic_setting`. Additional parameters are located in `src/experiments/toy_model.py`.

## Embedding Analysis (Section 6)

Final step embedding analysis is by default saved to output. To run additional embedding analysis, please use the `pca_plot` parameter in `src/experiments/toy_model.py`.

## Figure Generation

Figures for the naturalistic experiments were created in `src/notebooks/icl_bert.ipynb` and figures for the synthetic experiments were created in `src/notebooks/toy_model.ipnb`. 


The other scripts and files are useful for the pushdown experiments in the Appendix.


## References
Mitchell P. Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. "Building a Large Annotated Corpus of English: The Penn Treebank." *Computational Linguistics*, vol. 19, no. 2, June 1993, pp. 313-330. ISSN 0891-2017.
Yihong Chen, Kelly Marchisio, Roberta Raileanu, David Ifeoluwa Adelani, Pontus Stenetorp, Sebastian Riedel, and Mikel Artetxe. "Improving Language Plasticity via Pretraining with Active Forgetting," 2024.

