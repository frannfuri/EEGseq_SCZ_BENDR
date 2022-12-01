#!/bin/bash

# Train from scratch
python3 main_from_scratch.py linear --results-filename avp_pAug_bw_vpr_dp0307_f1f_th04 --dataset-directory ../BENDR_datasets/trivial_set --extra-aug --save-models --load-bendr-weigths --valid-per-record --freeze-first-layers
# Train LO/MSO from scratch
#python3 main.py linear --random-init --results-filename "new" --dataset-directory "datasets/decomp_study"
#python3 main.py BENDR --random-init --results-filename "new" --dataset-directory "datasets/decomp_study"

# Train LO/MSO from checkpoint
#python3 main.py linear --results-filename "new" --dataset-directory "datasets/decomp_study"
#python3 main.py BENDR --multi-gpu --results-filename "new" --dataset-directory "datasets/decomp_study_SA047"

# Train LO/MSO from checkpoint with frozen encoder
#python3 main.py linear --freeze-encoder --results-filename "new" --dataset-directory "datasets/decomp_study"
#python3 main.py BENDR --freeze-encoder --results-filename "new" --dataset-directory "datasets/decomp_study"
