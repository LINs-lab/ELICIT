datasets=( "arc_challenge" "bbh_boolean_expressions" "bbh_date_understanding" "bbh_reasoning_about_colored_objects" "bbh_temporal_sequences" "bbq_age" "boolq" "crows_pairs" "commonsense_qa" "ethics_commonsense" "ethics_justice" "glue_sst2" "hellaswag" "math_qa" "mmlu_pro_math" "openbookqa" "superglue_rte" "superglue_wic" "glue_mnli" "glue_qnli" "glue_cola" "bbq_religion" "deepmind" "mmlu_high_school_psychology" "bbh_logical_deduction_five_objects")

for task in "${datasets[@]}"; do
    echo "Process ${task}"
    python process_data.py --task $task
done