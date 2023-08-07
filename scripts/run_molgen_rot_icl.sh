# engines {azure/code-davinci-002, azure/gpt-35-turbo-0301, openai/gpt-4, google/models/text-bison-001}

export OPENAI_API_KEY=""
export AZURE_API_KEY=""
export PALM_API_KEY=""

python neural_lark/main_mol.py \
    --seed 3 \
    --dataset acrylates \
    --add_rule_instruction_flag \
    --num_samples 100 \
    --engine openai/gpt-4 \
    --rule_temperature 0.6 \
    --temperature 0.3 \
    --prompt_mode rot \
    --prompt_template wrule_and_mname \
    --freq_penalty 0.4 \
    --max_tokens 512 \
    --use_generic_grammar \
