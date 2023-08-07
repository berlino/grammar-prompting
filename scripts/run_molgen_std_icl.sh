# engines {azure/code-davinci-002, azure/gpt-35-turbo-0301, openai/gpt-4, google/models/text-bison-001}

export OPENAI_API_KEY=""
export AZURE_API_KEY=""
export PALM_API_KEY=""

python neural_lark/main_mol.py \
    --seed 1 \
    --dataset acrylates \
    --num_samples 100 \
    --engine openai/gpt-4 \
    --temperature 0.6 \
    --prompt_mode std \
    --prompt_template std \
    --freq_penalty 0.1 \
    --max_tokens 512 \