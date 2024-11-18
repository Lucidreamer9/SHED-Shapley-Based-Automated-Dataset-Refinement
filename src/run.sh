#!/bin/bash
# Create the "workspace" directory if it does not exist
if [ ! -d "workspace" ]; then
  mkdir workspace
  echo "Directory 'workspace' created."
else
  echo "Directory 'workspace' already exists."
fi

# Create the "output" directory if it does not exist
if [ ! -d "output" ]; then
  mkdir output
  echo "Directory 'output' created."
else
  echo "Directory 'output' already exists."
fi

# Create the "final_dataset" directory if it does not exist
if [ ! -d "final_dataset" ]; then
  mkdir final_dataset
  echo "Directory 'final_dataset' created."
else
  echo "Directory 'final_dataset' already exists."
fi
# Set working directory and paths
Number_of_clusters=3000
Number_of_finalset=5000
ORIGINAL_DATASET="path/to/original_dataset.txt"  

# Step 1: Run cluster_sen_tran.py to cluster the original dataset
echo "Clustering the original dataset..."
python cluster_sen_tran.py ${ORIGINAL_DATASET} ${Number_of_clusters}
python txt_json.py 


# Step 2: Run iteration_shapley.sh to prepare for Shapley value calculation
echo "Preparing for Shapley value calculation..."

for i in {1..20}; do
    clusternumber=${Number_of_clusters}
    data_path="./workspace/cluster_center_${clusternumber}_${i}.json"  
    count_fine_path="./workspace/count_file_${clusternumber}_${i}.txt"  
    temp_save_filepath="./workspace/randomout_${clusternumber}_${i}.json"  

    for j in {1..50}; do
        python -u finetune_fixseed.py \
            --base_model yahma/llama-7b-hf \
            --batch_size 128 \
            --cutoff_len 1024 \
            --micro_batch_size 8 \
            --num_epochs 3 \
            --add_lora True \
            --lora_r 128 \
            --lora_alpha 256 \
            --save_strategy no \
            --max_new_token 4 \
            --data_path "$data_path" \
            --dev_data_path "devdata_1444.jsonl" \  #If target on other tasks, pls use the corresponding eval set and modify the finetune_fixseed.py
            --verbose False \
            --resume_from_checkpoint None \
            --count_fine_path "$count_fine_path" \
            --output_dir "./output"

        python random_out.py \
            --out_num 60 \
            --out_filepath "$data_path" \
            --temp_save_filepath "$temp_save_filepath"
    done
done
#  Run calculate_s.py to calculate Shapley value
echo "Calculating Shapley values..."
python calculate_s.py ${Number_of_clusters}
echo "Sampling the final selected dataset..."
python sample_QOCS ${Number_of_clusters} ${Number_of_finalset}
python sample_QWCS ${Number_of_clusters} ${Number_of_finalset}
WORKSPACE_DIR="./workspace"
output_dir="./output"
rm -rf "$WORKSPACE_DIR"/*
rm -rf "$output_dir"/*
echo "Process completed."
