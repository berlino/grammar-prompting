# few-shot
# 16/16 for codex/gpt
# 8/8 for quickrun

# splits {indomain, comp}

# engines {azure/code-davinci-002, azure/gpt-35-turbo-0301, google/models/text-bison-001, openai/gpt-4}

export PALM_API_KEY=""

python neural_lark/main.py \
    --seed 1 \
    --dataset smc \
    --split_name comp \
    --num_shot 8 \
    --batch_size 8 \
    --engine google/models/text-bison-001 \
    --temperature 0.0 \
    --max_tokens 640 \
    --prompt_mode std \
    --prompt_template std \
    --retrieve_fn all \
    --quickrun \