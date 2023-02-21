#!/bin/bash

# Train from scratch
###python3 main_from_scratch.py linear --results-filename avp_pAug_pretOwn_vpr_dp0307_f1f_th04 --dataset-directory ../BENDR_datasets/decomp_study_SA010 --extra-aug --save-models --use-valid --valid-per-record --freeze-first-layers --own-init
#python3 main_from_scratch.py longlinear --results-filename avp_pAug_bw_dp0307_f1f_th04 --dataset-directory ../BENDR_datasets/h_scz_study --extra-aug --save-models --load-bendr-weigths --freeze-first-layers
#python3 main_from_scratch.py longlinear --results-filename avp_pAug_pretOwn_vpr_dp0507_f1f_th04 --dataset-directory ../BENDR_datasets/decomp_study_SA039 --extra-aug --save-models --own-init ./longlinear-rslts_avp_pAug_bw_dp0307_f1f_th04_len40ov30_/best_model_f0_h_scz_study_lr0.0001bs8.pt --use-valid --valid-per-record --freeze-first-layers

#python3 main_from_scratch.py linear classifier --results-filename avp_pAug_pretOwn_vpr_dp0507_f1f_th04_newL --dataset-directory ../BENDR_datasets/decomp_study_SA047 --extra-aug --save-models --own-init ../BENDR_datasets/linear-rslts_avp_pAug_bw_dp0307_f1f_th04_len40ov30_/best_model_f0_h_scz_study_lr0.0001bs8.pt --use-valid --valid-per-record --freeze-first-layers --loss-per-epoch
python3 main_from_scratch.py linear classifier --results-filename avp_pAug_pretOwn_vpr_dp0307_f1f_th04_bcePw02_stepLR01 --dataset-directory ../BENDR_datasets/decomp_study_SA007 --extra-aug --save-models  --own-init ../BENDR_datasets/linear-rslts_avp_pAug_bw_dp0307_f1f_th04_len40ov30_/best_model_f0_h_scz_study_lr0.0001bs8.pt --use-valid --valid-per-record --freeze-first-layers
#python3 main_from_scratch.py linear regressor --results-filename avp_pAug_pretOwn_vpr_dp0307_f1f_stepLR01_maeL --dataset-directory ../BENDR_datasets/decomp_study_SA039 --extra-aug --save-models  --own-init ../BENDR_datasets/linear-rslts_avp_pAug_bw_dp0307_f1f_th04_len40ov30_/best_model_f0_h_scz_study_lr0.0001bs8.pt --use-valid --valid-per-record --freeze-first-layers



#python3 main_from_scratch.py linear classifier --results-filename avp_pAug_bw_dp0307_f1f_th04_ce_o2 --dataset-directory ../BENDR_datasets/h_scz_study --extra-aug --save-models --load-bendr-weigths --freeze-first-layer --n-outputs 2

# Train LO/MSO from scratch
#python3 main.py linear --random-init --results-filename "new" --dataset-directory "datasets/decomp_study"
#python3 main.py BENDR --random-init --results-filename "new" --dataset-directory "datasets/decomp_study"

# Train LO/MSO from checkpoint
#python3 main.py linear --results-filename "new" --dataset-directory "datasets/decomp_study"
#python3 main.py BENDR --multi-gpu --results-filename "new" --dataset-directory "datasets/decomp_study_SA047"

# Train LO/MSO from checkpoint with frozen encoder
#python3 main.py linear --freeze-encoder --results-filename "new" --dataset-directory "datasets/decomp_study"
#python3 main.py BENDR --freeze-encoder --results-filename "new" --dataset-directory "datasets/decomp_study"

