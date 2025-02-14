datasets=( "arc_challenge" "bbh_boolean_expressions" "bbh_date_understanding" "bbh_reasoning_about_colored_objects" "bbh_temporal_sequences" "bbq_age" "boolq" "crows_pairs" "commonsense_qa" "ethics_commonsense" "ethics_justice" "glue_sst2" "hellaswag" "math_qa" "mmlu_pro_math" "openbookqa" "superglue_rte" "superglue_wic" "glue_mnli" "glue_qnli" "glue_cola" "bbq_religion" "deepmind" "mmlu_high_school_psychology" "bbh_logical_deduction_five_objects")
retrieve_method="retriever"


for seed in 42 10 100; do
    for recall in 0.8; do
        for d_name in "${datasets[@]}"
            do  
                        echo "Running ${d_name}"
                        python elicit.py \
                        --dataset_name ${d_name} \
                        --state_dir library_results/llama3 \
                        --model_name model_path \
                        --retrieve_method $retrieve_method \
                        --test_samples 100 \
                        --save_dir eval_results/llama3_seed$seed \
                        --weight_ori 1.0 \
                        --weight_fv 2.0 \
                        --dataset_split test \
                        --shots 16 \
                        --k 10 \
                        --single_layer_mode \
                        --recall $recall \
                        --local \
                        --prompt_file dataset_files/natural_prompts_manual.json \
                        --seed $seed \
                         --soft \
                         --recollect_state \
                         --retrieval_model prompt_classifier_model_feb12.pth \
                         --use_template
                done
        done
done