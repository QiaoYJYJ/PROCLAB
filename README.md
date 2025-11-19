# PROCLAB
This project builds a classification pipeline using constractive learning and bilinear attention network to predict DTI.

# Environment Setup
conda create -n PROCLAB

conda activate PROCLAB

# Model Training
python train_DTI.py --exp-id ExperimentName --config configs/bindingdb.yaml

The best-performing model will be automatically saved under:best_models/ExperimentName

# t-SNE Visualization
python vali.py \
    --protein_sequence "FMEPKFEFAVKFNAL......" \
    --vgfr_file figure3/data/PPARG_activate.csv \
    --decoy_file figure3/data/PPARG_decoy.csv \
    --model best_models/ExperimentName/best_models.pt \
    --outdir figure3/

# Violin Plot Visualization
python violin.py \
    --protein_sequence "LQLKLNHPESSQLFAKLLQKMTDLRQIVTEHVQLLQ" \
    --vgfr_file figure3/data/PPARG_activate.csv \
    --decoy_file figure3/data/PPARG_decoy.csv \
    --model best_models/ExperimentName/best_models.pt \
    --outdir figure3/
