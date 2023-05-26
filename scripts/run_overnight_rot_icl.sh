export PALM_API_KEY=""

python neural_lark/main.py \
    --seed 1 \
    --dataset overnight \
    --domain blocks \
    --num_shot 16 \
    --batch_size 16 \
    --engine google/models/text-bison-001 \
    --temperature 0.0 \
    --max_tokens 640 \
    --prompt_mode rot \
    --prompt_template wrule \
    --retrieve_fn all \
    --add_rule_instruction_flag \
    --lazy_constrain_flag \
    --quickrun \