export PALM_API_KEY=""

python neural_lark/main_mol.py \
    --seed 3 \
    --dataset acrylates \
    --add_rule_instruction_flag \
    --num_samples 100 \
    --engine azure/gpt-35-turbo-0301 \
    --rule_temperature 0.6 \
    --temperature 0.3 \
    --prompt_mode rot \
    --prompt_template wrule_and_mname \
    --freq_penalty 0.4 \
    --max_tokens 512 \
    --use_generic_grammar \
