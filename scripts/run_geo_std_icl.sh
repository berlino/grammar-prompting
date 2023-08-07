# engines {azure/code-davinci-002, azure/gpt-35-turbo-0301, openai/gpt-4, google/models/text-bison-001}

export OPENAI_API_KEY=""
export AZURE_API_KEY=""
export PALM_API_KEY=""

python neural_lark/main.py \
    --seed 1 \
    --dataset geoquery \
    --split_name iid_split \
    --num_shot 24 \
    --batch_size 24 \
    --engine  openai/gpt-4 \
    --temperature 0.0 \
    --max_tokens 256 \
    --prompt_mode std \
    --prompt_template std \
    --retrieve_fn all \
    --quickrun \
