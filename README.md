# CodeFast
Our code, model and data are released here.

## 1.Setup
```bash
bash setup.sh
mkdir log
```

## 2.Experiments on Code Llama-7B (at least 14GB GPU meomory required)

MBPP(baseline)
```bash
python main.py  --model codellama/CodeLlama-7b-hf  --tasks mbpp  --max_new_tokens 300  --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token   --use_comment --save_generations_path results/codellama_7b_mbpp_baseline --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_7b_mbpp_baseline.log
```

MBPP(CodeFast)
```bash
python main.py  --model codellama/CodeLlama-7b-hf  --tasks mbpp  --max_new_tokens 300  --additional_model models/GenGuard_Multi_PL_codellama_7b/model.pth --is_additional_model --do_sample True  --n_samples 1  --batch_size 1 --use_comment    --allow_code_execution   --save_generations  --use_auth_token   --precision bf16 --decoding_strategy greedy --save_generations_path results/codellama_7b_mbpp_codefast 2>&1|tee log/codellama_7b_mbpp_codefast.log
```


MBJSP(baseline)
```bash
python main.py  --model codellama/CodeLlama-7b-hf  --tasks mbxp  --language javascript --max_new_tokens 300  --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token  --save_generations_path results/codellama_7b_mbjsp_baseline --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_7b_mbjsp_baseline.log
```

MBJSP(CodeFast)
```bash
python main.py  --model codellama/CodeLlama-7b-hf  --tasks mbxp  --language javascript   --max_new_tokens 300  --additional_model models/GenGuard_Multi_PL_codellama_7b/model.pth --is_additional_model --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token    --save_generations_path results/codellama_7b_mbjsp_codefast --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_7b_mbjsp_codefast.log
```

MBGP(baseline)
```bash
python main.py  --model codellama/CodeLlama-7b-hf  --tasks mbxp  --language go --max_new_tokens 300  --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token  --save_generations_path results/codellama_7b_mbgp_baseline --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_7b_mbgp_baseline.log
```

MBGP(CodeFast)
```bash
python main.py  --model codellama/CodeLlama-7b-hf  --tasks mbxp  --language go  --max_new_tokens 300  --additional_model models/GenGuard_Multi_PL_codellama_7b/model.pth --is_additional_model --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token    --save_generations_path results/codellama_7b_mbgp_codefast --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_7b_mbgp_codefast.log
```

MBCPP(baseline)
```bash
python main.py  --model codellama/CodeLlama-7b-hf  --tasks mbxp  --language cpp --max_new_tokens 300  --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token  --save_generations_path results/codellama_7b_mbcpp_baseline --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_7b_mbcpp_baseline.log
```

MBCPP(CodeFast)
```bash
python main.py  --model codellama/CodeLlama-7b-hf  --tasks mbxp  --language cpp   --max_new_tokens 300  --additional_model models/GenGuard_Multi_PL_codellama_7b/model.pth --is_additional_model --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token    --save_generations_path results/codellama_7b_mbcpp_codefast --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_7b_mbcpp_codefast.log
```

## 3.Experiments on Code Llama-13B (at least 26GB GPU meomory required)

MBPP(baseline)
```bash
python main.py  --model codellama/CodeLlama-13b-hf  --tasks mbpp  --max_new_tokens 300  --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token   --use_comment --save_generations_path results/codellama_13b_mbpp_baseline --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_13b_mbpp_baseline.log
```

MBPP(CodeFast)
```bash
python main.py  --model codellama/CodeLlama-13b-hf  --tasks mbpp  --max_new_tokens 300  --additional_model models/GenGuard_Multi_PL_codellama_13b/model.pth --is_additional_model --do_sample True  --n_samples 1  --batch_size 1 --use_comment    --allow_code_execution   --save_generations  --use_auth_token   --precision bf16 --decoding_strategy greedy --save_generations_path results/codellama_13b_mbpp_codefast 2>&1|tee log/codellama_13b_mbpp_codefast.log
```


MBJSP(baseline)
```bash
python main.py  --model codellama/CodeLlama-13b-hf  --tasks mbxp  --language javascript --max_new_tokens 300  --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token  --save_generations_path results/codellama_13b_mbjsp_baseline --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_13b_mbjsp_baseline.log
```

MBJSP(CodeFast)
```bash
python main.py  --model codellama/CodeLlama-13b-hf  --tasks mbxp  --language javascript   --max_new_tokens 300  --additional_model models/GenGuard_Multi_PL_codellama_13b/model.pth --is_additional_model --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token    --save_generations_path results/codellama_13b_mbjsp_codefast --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_13b_mbjsp_codefast.log
```

MBGP(baseline)
```bash
python main.py  --model codellama/CodeLlama-13b-hf  --tasks mbxp  --language go --max_new_tokens 300  --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token  --save_generations_path results/codellama_13b_mbgp_baseline --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_13b_mbgp_baseline.log
```

MBGP(CodeFast)
```bash
python main.py  --model codellama/CodeLlama-13b-hf  --tasks mbxp  --language go  --max_new_tokens 300  --additional_model models/GenGuard_Multi_PL_codellama_13b/model.pth --is_additional_model --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token    --save_generations_path results/codellama_13b_mbgp_codefast --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_13b_mbgp_codefast.log
```

MBCPP(baseline)
```bash
python main.py  --model codellama/CodeLlama-13b-hf  --tasks mbxp  --language cpp --max_new_tokens 300  --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token  --save_generations_path results/codellama_13b_mbcpp_baseline --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_13b_mbcpp_baseline.log
```

MBCPP(CodeFast)
```bash
python main.py  --model codellama/CodeLlama-13b-hf  --tasks mbxp  --language cpp   --max_new_tokens 300  --additional_model models/GenGuard_Multi_PL_codellama_13b/model.pth --is_additional_model --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token    --save_generations_path results/codellama_13b_mbcpp_codefast --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_13b_mbcpp_codefast.log
```


## 4.Experiments on Code Llama-34B (at least 68GB GPU meomory required)

MBPP(baseline)
```bash
python main.py  --model codellama/CodeLlama-34b-hf  --tasks mbpp  --max_new_tokens 300  --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token   --use_comment --save_generations_path results/codellama_34b_mbpp_baseline --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_34b_mbpp_baseline.log
```

MBPP(CodeFast)
```bash
python main.py  --model codellama/CodeLlama-34b-hf  --tasks mbpp  --max_new_tokens 300  --additional_model models/GenGuard_Multi_PL_codellama_34b/model.pth --is_additional_model --do_sample True  --n_samples 1  --batch_size 1 --use_comment    --allow_code_execution   --save_generations  --use_auth_token   --precision bf16 --decoding_strategy greedy --save_generations_path results/codellama_34b_mbpp_codefast 2>&1|tee log/codellama_34b_mbpp_codefast.log
```


MBJSP(baseline)
```bash
python main.py  --model codellama/CodeLlama-34b-hf  --tasks mbxp  --language javascript --max_new_tokens 300  --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token  --save_generations_path results/codellama_34b_mbjsp_baseline --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_34b_mbjsp_baseline.log
```

MBJSP(CodeFast)
```bash
python main.py  --model codellama/CodeLlama-34b-hf  --tasks mbxp  --language javascript   --max_new_tokens 300  --additional_model models/GenGuard_Multi_PL_codellama_34b/model.pth --is_additional_model --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token    --save_generations_path results/codellama_34b_mbjsp_codefast --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_34b_mbjsp_codefast.log
```

MBGP(baseline)
```bash
python main.py  --model codellama/CodeLlama-34b-hf  --tasks mbxp  --language go --max_new_tokens 300  --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token  --save_generations_path results/codellama_34b_mbgp_baseline --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_34b_mbgp_baseline.log
```

MBGP(CodeFast)
```bash
python main.py  --model codellama/CodeLlama-34b-hf  --tasks mbxp  --language go  --max_new_tokens 300  --additional_model models/GenGuard_Multi_PL_codellama_34b/model.pth --is_additional_model --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token    --save_generations_path results/codellama_34b_mbgp_codefast --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_34b_mbgp_codefast.log
```

MBCPP(baseline)
```bash
python main.py  --model codellama/CodeLlama-34b-hf  --tasks mbxp  --language cpp --max_new_tokens 300  --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token  --save_generations_path results/codellama_34b_mbcpp_baseline --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_34b_mbcpp_baseline.log
```

MBCPP(CodeFast)
```bash
python main.py  --model codellama/CodeLlama-34b-hf  --tasks mbxp  --language cpp   --max_new_tokens 300  --additional_model models/GenGuard_Multi_PL_codellama_34b/model.pth --is_additional_model --do_sample True  --n_samples 1  --batch_size 1   --allow_code_execution   --save_generations  --use_auth_token    --save_generations_path results/codellama_34b_mbcpp_codefast --precision bf16 --decoding_strategy greedy 2>&1|tee log/codellama_34b_mbcpp_codefast.log
```


