export PALM_API_KEY=""

python neural_lark/main_mol.py \
    --seed 1 \
    --dataset acrylates \
    --num_samples 100 \
    --engine azure/code-davinci-002 \
    --temperature 0.6 \
    --prompt_mode std \
    --prompt_template std \
    --freq_penalty 0.1 \
    --max_tokens 512 \