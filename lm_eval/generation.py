import json
from math import ceil

from accelerate.utils import set_seed
from torch.utils.data.dataloader import DataLoader
from transformers import StoppingCriteria, StoppingCriteriaList
import os
from lm_eval.utils import TokenizedDataset, complete_code
import torch
import mxeval
from tqdm import tqdm
from mxeval.evaluate_functional_correctness import entry_point
from mxeval.data import write_jsonl
import time
class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""
    def __init__(self, start_length, eof_strings, tokenizer, check_fn=None):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        if check_fn is None:
            check_fn = lambda decoded_generation: any(
                [stop_string in decoded_generation for stop_string in self.eof_strings]
            )
        self.check_fn = check_fn

    def __call__(self, input_ids, scores, **kwargs):
        
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        return all([self.check_fn(decoded_generation) for decoded_generation in decoded_generations])

class TooLongFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if the generated function is too long by a certain multiplier based on input length."""

    def __init__(self, input_length, multiplier):
        self.input_length = input_length
        self.multiplier = multiplier

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if generated sequence is too long."""
        return input_ids.shape[1] > int(self.input_length * self.multiplier)
        

def parallel_generations(task, dataset, accelerator, model, tokenizer, n_tasks, args):
    if args.load_generations_path:
        # load generated code
        with open(args.load_generations_path) as fp:
            generations = json.load(fp)
            if accelerator.is_main_process:
                print(
                    f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
                )
        return generations[:n_tasks]

    set_seed(args.seed, device_specific=True)

    # Setup generation settings
    if args.decoding_strategy =='greedy':
        gen_kwargs = {
            # "do_sample": args.do_sample,
            "temperature": args.temperature,
            'num_return_sequences':args.batch_size,
            # "top_p": args.top_p,
            # "top_k": args.top_k,
            # "max_length": args.max_length_generation,
        }
    else:
        gen_kwargs = {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            'num_return_sequences':args.batch_size,
            # "max_length": args.max_length_generation,
        }
    # stopping_criteria = []
    # # The input_length / start_length set to 0 for now will be adjusted later
    # # Check if the task has a custom check_fn method for the stopping criteria
    # if task.stop_words and tokenizer.eos_token:
    #     task.stop_words=[tokenizer.eos_token]   
    # if hasattr(task, "check_fn"):
    #     stopping_criteria.append(
    #         EndOfFunctionCriteria(0, task.stop_words, tokenizer, task.check_fn)
    #     )
    # elif task.stop_words:
    #     stopping_criteria.append(
    #         EndOfFunctionCriteria(0, task.stop_words, tokenizer)
    #     )
    # if hasattr(task, "max_length_multiplier") and task.max_length_multiplier:
    #     stopping_criteria.append(
    #         TooLongFunctionCriteria(0, task.max_length_multiplier)
    #     )
    
    # if stopping_criteria:
    #     gen_kwargs["stopping_criteria"] = StoppingCriteriaList(stopping_criteria)

    if args.instruction_tokens:
        instruction_tokens = args.instruction_tokens.split(",")
        if len(instruction_tokens) != 3:
            raise ValueError(
                "Instruction tokens should contain exactly 3 tokens separated by a comma. If a token is empty, represent it as ''"
            )
        for token in instruction_tokens:
            if token.strip() != "":
                task.stop_words.append(token)
    else:
        instruction_tokens = None
    if accelerator.is_main_process:
        print(f"number of problems for this task is {n_tasks}")
    n_copies = ceil(args.n_samples / args.batch_size)

    ds_tokenized = TokenizedDataset(
        task,
        dataset,
        tokenizer,
        num_devices=accelerator.state.num_processes,
        max_length=args.max_length_generation,
        limit_start=args.limit_start,
        n_tasks=n_tasks,
        n_copies=n_copies,
        prefix=args.prefix,
        has_encoder=args.modeltype == "seq2seq",
        instruction_tokens=instruction_tokens,
    )

    # do not confuse args.batch_size, which is actually the num_return_sequences
    ds_loader = DataLoader(ds_tokenized, batch_size=1)

    is_loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    is_loaded_in_4bit = getattr(model, "is_loaded_in_4bit", False)
    if args.max_memory_per_gpu is not None:
        # The model is already sharded across multiple GPUs
        ds_loader = accelerator.prepare(ds_loader)
    elif not is_loaded_in_8bit and not is_loaded_in_4bit:
        # we only wrap data loader to avoid extra memory occupation
        model = model.to(accelerator.device)
        ds_loader = accelerator.prepare(ds_loader)
    else:
        # model.to() is not supported for 8bit and 4bit models
        model, ds_loader = accelerator.prepare(model, ds_loader)

    generations = complete_code(
        task,
        accelerator,
        model,
        tokenizer,
        ds_loader,
        n_tasks=n_tasks,
        limit_start=args.limit_start,
        batch_size=args.batch_size,
        prefix=args.prefix,
        instruction_tokens=instruction_tokens,
        postprocess=args.postprocess,
        is_wrapped=is_loaded_in_8bit or is_loaded_in_4bit,
        **gen_kwargs,
    )
    return generations

def normal_generations(task, dataset, accelerator, model, tokenizer, n_tasks, args):
    if args.load_generations_path:
        # load generated code
        with open(args.load_generations_path+'/generations.json') as fp:
            generations = json.load(fp)
            if accelerator.is_main_process:
                print(
                    f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
                )
        return generations[:n_tasks]

    set_seed(args.seed, device_specific=True)

    # Setup generation settings
    if args.decoding_strategy =='greedy':
        gen_kwargs = {}
    else:
        gen_kwargs = {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            'num_return_sequences':args.batch_size,
            # "max_length": args.max_length_generation,
        }
    stopping_criteria = []
    # The input_length / start_length set to 0 for now will be adjusted later
    # Check if the task has a custom check_fn method for the stopping criteria
    # if task.stop_words and tokenizer.eos_token:
    #     task.stop_words.append(tokenizer.eos_token)    
    #     task.stop_words = []
    # if task.stop_words and tokenizer.eos_token:
    #     task.stop_words=[tokenizer.eos_token]  
    # print(task.stop_words)
    # # input()
    # if hasattr(task, "check_fn"):
    #     stopping_criteria.append(
    #         EndOfFunctionCriteria(0, task.stop_words, tokenizer, task.check_fn)
    #     )
    # elif task.stop_words:
    #     stopping_criteria.append(
    #         EndOfFunctionCriteria(0, task.stop_words, tokenizer)
    #     )
    # if hasattr(task, "max_length_multiplier") and task.max_length_multiplier:
    #     stopping_criteria.append(
    #         TooLongFunctionCriteria(0, task.max_length_multiplier)
    #     )
    
    # if stopping_criteria:
    #     gen_kwargs["stopping_criteria"] = StoppingCriteriaList(stopping_criteria)

    if args.instruction_tokens:
        instruction_tokens = args.instruction_tokens.split(",")
        if len(instruction_tokens) != 3:
            raise ValueError(
                "Instruction tokens should contain exactly 3 tokens separated by a comma. If a token is empty, represent it as ''"
            )
        for token in instruction_tokens:
            if token.strip() != "":
                task.stop_words.append(token)
    else:
        instruction_tokens = None
    if accelerator.is_main_process:
        print(f"number of problems for this task is {n_tasks}")
    n_copies = ceil(args.n_samples / args.batch_size)
 

    is_loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    is_loaded_in_4bit = getattr(model, "is_loaded_in_4bit", False)

        

    if args.max_memory_per_gpu is not None:
        # The model is already sharded across multiple GPUs
        ds_loader = accelerator.prepare(ds_loader)
    elif not is_loaded_in_8bit and not is_loaded_in_4bit:
        # we only wrap data loader to avoid extra memory occupation
        model = model.to(accelerator.device)
        # ds_loader = accelerator.prepare(ds_loader)
    else:
        # model.to() is not supported for 8bit and 4bit models
        model, ds_loader = accelerator.prepare(model, ds_loader)
    gen_codes=[]
    gen_codes = [[] for _ in range(n_tasks)]
    args.limit_start = 0
    mbxp_sample_list = []
    prompt_list = []
    
    def load_additional_model(model_path,model_type):
        if model_type == 'linear':
            # Load saved state_dict
            state_dict = torch.load(args.additional_model)
            # Get in_features and out_features from state_dict
            in_features = state_dict['weight'].size(1)
            out_features = state_dict['weight'].size(0)
            # Create the linear model with the retrieved dimensions
            model = torch.nn.Linear(in_features, out_features)
            # Load the state_dict into the model
            model.load_state_dict(state_dict)
            # Convert the model to bfloat16
            model = model.to(dtype=torch.bfloat16)
            # Move to GPU if available
            model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        return model


    #setup CodeFast config
    if args.is_additional_model:
        codefast_config_dict={}
        #load genguard model
        additional_model = load_additional_model(args.additional_model,'linear')

        # # The token represents the end of a code line
        if 'starcoder' in args.model:
            code_line_end_token = 203
        elif 'incoder' in args.model:
            code_line_end_token  = 205
        else:
            code_line_end_token  = 13
        
        codefast_config_dict['additional_model'] = additional_model
        codefast_config_dict['stop_threshold'] = args.stop_threshold
        codefast_config_dict['additional_model_type'] ='linear'
        codefast_config_dict['code_line_end_token'] = code_line_end_token
        codefast_config_dict['continue_label'] = args.continue_label
        codefast_config_dict['is_convergence'] = args.is_convergence
        codefast_config_dict['convergence_num'] = args.convergence_num


    start_time = time.time()
    for sample in tqdm(range(args.limit_start, args.limit_start + n_tasks)):

        # infill = []
        # instruction = []
        torch.cuda.reset_max_memory_allocated()
        if args.tasks == 'mbpp':
            
            if args.start_retrieval_aug == False:
                prompt_contents = task.get_prompt(dataset[sample],n_shot =args.n_shot)
            else:
                prompt_contents = task.get_prompt(dataset[sample],n_shot =args.n_shot,start_retrieval_aug =True)

        else:
            prompt_contents = task.get_prompt(dataset[sample])

        if isinstance(prompt_contents, str):
            prompt = args.prefix + prompt_contents

        prompt_list.append(prompt)
        prompt_input = [prompt]
        batch = tokenizer(
        prompt_input,
        return_tensors="pt",
        return_token_type_ids=False,
        )  

        
        batch = {k: v.to("cuda") for k, v in batch.items()}
        is_out_of_memory = False

        max_new_tokens = args.max_new_tokens


        try:

            if args.is_additional_model:
                
                with torch.no_grad():
                    outputs = model.generate(
                        **batch,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=10,
                        
                        codefast_config_dict = codefast_config_dict,
                        
                        **gen_kwargs 
                    )


              
            else:
                
                with torch.no_grad():
                    outputs = model.generate(
                        **batch,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=10,
                        
                        **gen_kwargs 
                    )

        except (torch.cuda.OutOfMemoryError) as e:
            output_text = ['<out_of_memory>']
            is_out_of_memory = True

        sample_list = []
        if not is_out_of_memory :
            for o in outputs:
                output_text = tokenizer.decode(o, skip_special_tokens=True)  
                if args.postprocess:
                    gen_codes[sample].append(
                        task.postprocess_generation(output_text, int(sample) + args.limit_start)
                    )
                else:
                    print(
                        "model output is not postprocessed, this might lower evaluation scores"
                    )
                    
                    gen_codes[sample].append(output_text)
                if 'mbxp' in args.tasks or 'humanevalx' in args.tasks:
                    mbxp_sample_list.append(dict(task_id=dataset[sample]['task_id'], language=dataset[sample]["language"], completion=task.postprocess_generation(output_text, int(sample) + args.limit_start).replace(prompt,'')))
          
        else:
            gen_codes[sample].append(output_text)
            if 'mbxp' in args.tasks or 'humanevalx' in args.tasks:
                    mbxp_sample_list.append(dict(task_id=dataset[sample]['task_id'], language=dataset[sample]["language"], completion='<out_of_memory>'))
               
    end_time = time.time()
    avg_time = (end_time-start_time)/n_tasks
    if not os.path.exists(args.save_generations_path):
            # 如果路径不存在，创建文件夹
        os.makedirs(args.save_generations_path)
    final_ans_dir = args.save_generations_path+'/evaluation_results.json'
    with open(final_ans_dir,'w') as f:
        json.dump({'average_time':avg_time},f)
    if 'mbxp' in args.tasks or 'humanevalx' in args.tasks:
        write_jsonl(args.save_generations_path+"/samples.jsonl", mbxp_sample_list)
        if 'mbxp' in args.tasks:
            problem_path_dict={
                'go':'mbgp_release_v1.1.jsonl',
                'cpp': 'mbcpp_release_v1.2.jsonl',
                'java': 'mbjp_release_v1.2.jsonl',
                'javascript':'mbjsp_release_v1.2.jsonl',
                'python': 'mbpp_release_v1.jsonl'
            }
            problem_file = 'mxeval/data/mbxp/'+problem_path_dict[args.language]
        elif 'humanevalx' in args.tasks:
            problem_path_dict={
            'go':'HumanEval_go_v1.jsonl',
            'javascript':'HumanEval_javascript_v1.1.jsonl',
            'python': 'HumanEval.jsonl'
            }
            problem_file = 'mxeval/data/multilingual_humaneval/'+problem_path_dict[args.language]

        sample_file = args.save_generations_path+"/samples.jsonl" 
        entry_point(sample_file,problem_file)
        with open(sample_file+'_passatk.json','r') as f:
            mxeval_result = json.load(f)
        with open(final_ans_dir,'r') as f:
            evaluation_results = json.load(f)
        evaluation_results.update(mxeval_result)
        with open(final_ans_dir,'w') as f:
            json.dump(evaluation_results,f)
        
        
    return gen_codes