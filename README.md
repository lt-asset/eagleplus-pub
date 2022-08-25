# EAGLE

## Folder/File Structure

The `EAGLE` directory contains codes for reproducing our experiments.

Directory `newrules` contains code for the new rules for TorchRec and other new applications. 

## Instruction

### Create environment
To create the docker container, run `bash EAGLE/docker_command`.

### Generate input datasets
To generate and save input datasets, under the `EAGLE` directory, run `python -m newrules.gen_torchrec_model_and_dataset`. Input will be saved under `EAGLE/data/dataset` and models will be saved under `EAGLE/data/models`.

### Execute EAGLEPlus rules
To execute the new rules regarding distributed versus non-distributed training and inference (e.g., rule 17), run `bash execute_testing_new_rules.sh`. The outputs will be saved under `EAGLE/data/outputs`.

### Analyze results
After execution, use `python analyze_results_distributed.py` to analyze the results for the new rules regarding distributed versus non-distributed training and inference (e.g., rule 17).
