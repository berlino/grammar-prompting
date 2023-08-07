# engines {azure/code-davinci-002, azure/gpt-35-turbo-0301, openai/gpt-4, google/models/text-bison-001}

export OPENAI_API_KEY=""
export AZURE_API_KEY=""
export PALM_API_KEY=""

python neural_lark/main.py \
    --seed 1 \
    --dataset overnight \
    --domain blocks \
    --num_shot 16 \
    --batch_size 16 \
    --engine openai/gpt-4 \
    --temperature 0.0 \
    --max_tokens 640 \
    --prompt_mode rot \
    --prompt_template wrule \
    --retrieve_fn all \
    --add_rule_instruction_flag \
    --lazy_constrain_flag \
    --quickrun \