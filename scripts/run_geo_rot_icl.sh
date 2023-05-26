# few-shot
# 32/32 for codex/gpt
# 24/24 for quickrun

# many-shot
# -1/42 for codex/gpt

# splits {iid_split, template_split, tmcd_split, length_split, zero_shot}

# --constrain_rule_gen_flag \
# --constrain_prog_gen_flag \

export PALM_API_KEY=""

python neural_lark/main.py \
    --seed 1 \
    --dataset geoquery \
    --split_name iid_split \
    --num_shot 24 \
    --batch_size 24 \
    --engine google/models/text-bison-001  \
    --temperature 0.0 \
    --max_tokens 256 \
    --prompt_mode rot \
    --prompt_template wrule \
    --retrieve_fn all \
    --add_rule_instruction_flag \
    --lazy_constrain_flag \
    --quickrun \
