datasets=("boolq" "crows_pairs" "commonsense_qa" "ethics_commonsense" "ethics_justice" "glue_sst2" "hellaswag" "math_qa" "mmlu_pro_math" "openbookqa" "superglue_rte" "superglue_wic" "glue_mnli" "glue_qnli" "glue_cola" "bbq_religion" "deepmind" "mmlu_high_school_psychology" "bbh_logical_deduction_five_objects")


for d_name in "${datasets[@]}"; do
        echo "Running Script for: ${d_name} 16 shots"
        python compute_avg_hidden_state.py \
        --dataset_name="${d_name}"\
        --save_path_root="library_results/llama3" \
        --model_name='model_path' \
        --dataset_split valid \
        --n_shots 16 \
        --model_dtype 16 \
        --save_icl \
        --test_nums 100 \
        --fluency \
        --recompute \
        --eval \
        --weight_ori 1.0 \
        --weight_fv 2.0 
done