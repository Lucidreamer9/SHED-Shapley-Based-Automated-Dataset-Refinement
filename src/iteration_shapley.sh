
# This script is used to generate the files for the shapley value calculation.
for i in {1..10}; do
    clusternumber=3000
    data_path="path/to/cluster_center_${clusternumber}_${i}.json" #copy the cluster center data to several files, and change the index, to iterative calculate the marginal contribution
    count_fine_path="path/to/count_file_${clusternumber}_${i}.txt" #store the accuracy of the cluster center data
    temp_save_filepath="path/to/randomout_${clusternumber}_${i}.json" #store the random out data
    for i in {1..50}
    do
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
        --dev_data_path "path/to/dev_data" \
        --verbose False \
        --resume_from_checkpoint "path/to/initial_checkpoint"\
        --count_fine_path "$count_fine_path" \
        --output_dir "path/to/outoput"
        
    python random_out.py\
        --out_num 60\
        --out_filepath "$data_path" \
        --temp_save_filepath "$temp_save_filepath" 
    
    
    
    done
done
