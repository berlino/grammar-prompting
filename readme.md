## Setup

Basic Setup

```bash
conda create --name grammar-prompting
pip install -e .
pip install git+https://github.com/yuce/pyswip@master#egg=pyswip
```

### LLM setup

 In prompting scripts, you can use the following strings to specify which LLM to use: `azure/code-davinci-002, azure/gpt-35-turbo-0301, openai/gpt-4, google/models/text-bison-001`. You should also provide corresponding API keys in the scripts. Our original experiments are done via GPT APIs provided by Azure, from which you can still get access to Codex (WARNING: super expensive). 

By default, the scoring function for constrained decoding is based on sentence BERT. If your setup has access to Codex, you can comment out these lines to activate Codex-based scoring. 

### Setup for molecule generation experiments

For molecule generation experiments, the following setups are additionally required.

```bash
pip install rdkit
pip install -e third_party/retro_star/retro_star/packages/mlp_retrosyn/
pip install -e third_party/retro_star/retro_star/packages/rdchiral/
pip install -e third_party/fuseprop/
pip install -e third_party/GCN
```


## Code structure and scripts

`neural_lark` contains code for handling data, prompting and constrained generation.
`minEarley` contains the code for Earley-based parsing. The parser code is adapted from [Lark](https://github.com/lark-parser/lark). 

### Semantic Parsing

```bash
run_geo_std_icl.sh  # standard prompting on geoqeury
run_geo_rot_icl.sh  # grammar prompting on geoquery
run_overnight_std_icl.sh # standard prompting on Overnight-Block
run_overnight_cot_icl.sh # grammar prompting on Overnight-Block
run_smc_std_icl.sh # standard prompting on SMC
run_smc_cot_icl.sh # grammar prompting on SMC
```

Here is the link to [LLM cache](https://drive.google.com/file/d/1cUPnWE6x3TvlDs-rtn6oGhp0cd1ludnw/view?usp=sharing), which you can download and place under the root directory. With the cache in place, the scripts above should reproduce the results in Table 3.

### Molecule Generation

```bash
run_molgen_std_icl.sh # standard prompting on molecule generation
run_molgen_cot_icl.sh # grammar prompting on molecule generation
```

The results reported in the paper are obtained via `azure/gpt-35-turbo-0301`, though the current scripts use `openai/gpt-4` for the setup without Azure.