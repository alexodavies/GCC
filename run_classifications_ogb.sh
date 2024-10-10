#!/bin/bash

# List of datasets to iterate over
datasets=("ogbg-molclintox"  "ogbg-molbbbp" "ogbg-molbace" "ogbg-molhiv")

# Training parameters
gpu=0
load_path="saved/moco/current.pth"
epochs=50


# Loop through each dataset and run the command
for dataset in "${datasets[@]}"
do
    echo "Training on dataset: $dataset"
    python -W ignore train_supervised_ogb_with_pos_encodings.py --dataset $dataset --gpu $gpu --load-path $load_path --epochs $epochs
    echo "Finished training on dataset: $dataset"
done

load_path="saved/no_moco/current.pth"
# Loop through each dataset and run the command
for dataset in "${datasets[@]}"
do
    echo "Training on dataset: $dataset"
    python -W ignore train_supervised_ogb.py --dataset $dataset --gpu $gpu --load-path $load_path --epochs $epochs
    echo "Finished training on dataset: $dataset"
done



