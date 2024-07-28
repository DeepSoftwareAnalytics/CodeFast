import re

from evaluate import load
import json
from lm_eval.base import Task

class HumanEvalX(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # DATASET_PATH = 'mxeval/mbxp'
    DATASET_NAME = None
    def __init__(self,language="java"):
        self.DATASET_NAME = language
        if language == 'go':
            stop_words = ['\nfunc']
        elif language == 'javascript':
            stop_words = ['\nconsole','/**\n * * ','\n/**','\nif (require','\nmodule']
        elif language == 'cpp':
            stop_words = ['\nint main','/**\n * Write'] 
        elif language =='java':
            stop_words = ['\n     public static void']
        elif language =='python':
            stop_words=["\ndef","\nif","\nassert","\nclass",'\n"""', "\nprint", "\n<|/","\n#", "\n@"]
        else:
            stop_words  =[]
        problem_path_dict={
            'go':'mxeval/data/multilingual_humaneval/HumanEval_go_v1.jsonl',
            'cpp': 'mxeval/data/multilingual_humaneval/mbcpp_release_v1.2.jsonl',
            'java': 'mxeval/data/multilingual_humaneval/mbjp_release_v1.2.jsonl',
            'javascript':'mxeval/data/multilingual_humaneval/HumanEval_javascript_v1.1.jsonl',
            'python': 'mxeval/data/multilingual_humaneval/HumanEval.jsonl'
        }
        super().__init__(
            
            stop_words=stop_words,
            #"\ndef","\nassert","\nclass",'\n"""', "\nprint", "\nif", "\n<|/","\n#", "\n@"
            requires_execution=True,
        )
        # self.dataset = self.dataset["test"]
        self.dataset_dir = problem_path_dict[self.DATASET_NAME]
        # self.demonstration = None
        # self.n_shot =None
        # self.current_func_name =None
        # self.start_retrieval_aug =False
        # self.comment = ''
        # self.is_requirement_before = False

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        MBPP prompt is built following to InCoder (Fried et al.) approach
        prompt = docstring that includes one test
        """
        prompt = doc['prompt']
        return prompt

    def get_dataset(self,use_train=False):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        # for d in range(len(self.dataset)):
        #     print(self.dataset[d])
        #     input()
        
        with open(self.dataset_dir, "r") as f:
            self.dataset = [json.loads(line) for line in f if line.strip()]
        
        return self.dataset
            # the wrong split of mbpp can be loaded with old datasets cache
            # assert (    
            #     len(dataset) == 500
            # ), "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        
    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["test"]

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]
    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        
        
        # if self.n_shot!=None:
        #     prompt = self.get_prompt(self.dataset["test"][idx],n_shot =  self.n_shot,start_retrieval_aug=self.start_retrieval_aug)

        #     generation = generation[len(prompt) :]
        #     return  self.current_func_name + self._stop_at_stop_token(generation, self.stop_words)
        prompt = self.get_prompt(self.dataset[idx])
        generation = generation[len(prompt) :]
        # generation =  generation.replace('\t','    ')
        return prompt+ self._stop_at_stop_token(generation, self.stop_words)
        
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_metric = load("code_eval")
        results, test_case_results  = code_metric.compute(
            references=references,
            predictions=generations,
            k=[1,3,5]
        )
        return results, test_case_results

        # results, test_case_results = code_metric.compute(
        #     references=references,
        #     predictions=generations,
        #     language=self.DATASET_NAME,
        #     timeout=15.0,
        #     num_workers=1,
        # )
        # return results, test_case_results