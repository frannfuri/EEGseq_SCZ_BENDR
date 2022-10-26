#!/bin/bash

# Train from scratch
python3 main_from_scratch.py --results-filename probAug_linf_vpr_th06_freeze3first --dataset-directory ./datasets/h_scz_study --use-valid --extra-aug --valid-per-record --save-models --load-bendr-weigths --freeze-first-layers
# Train LO/MSO from scratch
#python3 main.py linear --random-init --results-filename "new" --dataset-directory "datasets/decomp_study"
#python3 main.py BENDR --random-init --results-filename "new" --dataset-directory "datasets/decomp_study"

# Train LO/MSO from checkpoint
#python3 main.py linear --results-filename "new" --dataset-directory "datasets/decomp_study"
#python3 main.py BENDR --multi-gpu --results-filename "new" --dataset-directory "datasets/decomp_study_SA047"

# Train LO/MSO from checkpoint with frozen encoder
#python3 main.py linear --freeze-encoder --results-filename "new" --dataset-directory "datasets/decomp_study"
#python3 main.py BENDR --freeze-encoder --results-filename "new" --dataset-directory "datasets/decomp_study"

