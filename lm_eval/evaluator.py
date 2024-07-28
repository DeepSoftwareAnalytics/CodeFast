import inspect
import json
import os
import warnings

from lm_eval import tasks
from lm_eval.generation import parallel_generations, normal_generations

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""


class Evaluator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # setup arguments
        self.metric_output_path = args.metric_output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

    def generate_text(self, task_name):
        task = tasks.get_task(task_name, self.args)
        if 'codellama' in self.args.model and self.args.tasks=='mbpp':
            use_comment = True
        else:
            use_comment = self.args.use_comment 
        is_requirement_before = self.args.is_requirement_before
        if self.args.tasks == 'mbpp':
            dataset = task.get_dataset(use_train = self.args.use_train, use_comment = use_comment, is_requirement_before = self.args.is_requirement_before)
      
        elif self.args.tasks == 'mbxp':
            dataset = task.get_dataset(use_train = self.args.use_train)
        else :
            dataset = task.get_dataset()
        

        # if args.limit is None, use all samples
        n_tasks = self.args.limit if self.args.limit else len(dataset)
        references = [task.get_reference(dataset[i]) for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]
        
       
        if self.args.check_references:
            if "get_solution" in inspect.signature(task.get_reference).parameters:
                solutions = [[task.get_reference(dataset[i], get_solution=True)] for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references


        generations = normal_generations(
            task,
            dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args,
        )

        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations, references

    def evaluate(self, task_name):
        task = tasks.get_task(task_name, self.args)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        generations, references = self.generate_text(task_name)

        if self.accelerator.is_main_process:
            if not self.args.load_generations_path:
                if self.args.save_generations:
                    
                    if not os.path.exists(self.args.save_generations_path):
                    # 如果路径不存在，创建文件夹
                        os.makedirs(self.args.save_generations_path)
                    save_dir  = self.args.save_generations_path+'/generations.json'
                   
                    with open(save_dir, "w") as fp:
                        json.dump(generations, fp)
                        print(
                            f"generations were saved at {self.args.save_generations_path}"
                        )
                # if self.args.save_references:
                #     with open("references.json", "w") as fp:
                #         json.dump(references, fp)
                #         print("references were saved at references.json")

            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            print("Evaluating generations...")
            results,test_case_results = task.process_results(generations, references)
            if self.args.load_generations_path != None:
                test_case_dir = self.args.load_generations_path+'/test_case.json'
            else:
                test_case_dir = self.args.save_generations_path+'/test_case.json'
            test_case_dict = {'metrics':results,'test_case':test_case_results,'generations':generations}
            final_ans_dir = self.args.save_generations_path+'/evaluation_results.json'
            if 'mbxp' not in  self.args.tasks and 'humanevalx' not in self.args.tasks:
                with open(final_ans_dir,'r') as f:
                    evaluation_results = json.load(f)
                evaluation_results.update(results)
                with open(final_ans_dir,'w') as f:
                    json.dump(evaluation_results,f)
            with open(test_case_dir,'w')as f:
                json.dump(test_case_dict,f)
            return results,test_case_results
