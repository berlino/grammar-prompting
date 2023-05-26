# few-shot
# 32/32 for codex/gpt
# 32/32 for palm

# many-shot
# -1/42 for codex/gpt

# splits {iid_split, template_split, tmcd_split, length_split, zero_shot}

# engines {azure/code-davinci-002, azure/gpt-35-turbo-0301, google/models/text-bison-001, openai/gpt-4}

export PALM_API_KEY=""

python neural_lark/main.py \
    --seed 1 \
    --dataset geoquery \
    --split_name iid_split \
    --num_shot 24 \
    --batch_size 24 \
    --engine google/models/text-bison-001 \
    --temperature 0.0 \
    --max_tokens 256 \
    --prompt_mode std \
    --prompt_template std \
    --retrieve_fn all \
    --quickrun \
