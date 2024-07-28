"""Program Synthesis with Large Language Models
https://arxiv.org/abs/2108.07732

The benchmark consists of around 1,000 crowd-sourced Python programming problems, 
designed to be solvable by entry level programmers, covering programming fundamentals, 
standard library functionality, and so on. Each problem consists of a task description, 
code solution and 3 automated test cases. As described in the paper, a subset of the data
has been hand-verified by the authors.

Homepage:: https://github.com/google-research/google-research/tree/master/mbpp
"""

import re

from evaluate import load
import json
from lm_eval.base import Task

_CITATION = """
@article{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
}
"""


class MBPP(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "mbpp"

    def __init__(self):
        super().__init__(
            
            stop_words=["\ndef","\nassert","\nclass",'\n"""', "\nprint", "\nif", "\n<|/","\n#", "\n@",],
            requires_execution=True,
        )
        self.demonstration = None
        self.n_shot =None
        self.current_func_name =None
        self.start_retrieval_aug =False
        self.comment = ''
        self.is_requirement_before = False
    def get_demonstration(self,n_shot =None):
        return ''.join(self.demonstration[:n_shot])
    def get_dataset(self,use_train=False,use_comment=False,is_requirement_before=False):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        if use_comment:
            self.comment = '    # Your code here\n'
        self.is_requirement_before = is_requirement_before
        if use_train:
            dataset = self.dataset["train"]
        else:    
            dataset = self.dataset["test"]
            # the wrong split of mbpp can be loaded with old datasets cache
            assert (
                len(dataset) == 500
            ), "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        demonstration_list = self.dataset['prompt']
        demonstration = []
        for doc_idx in range(1,len(demonstration_list)):
            doc = demonstration_list[doc_idx]
            
            description = doc["text"]
            test_example = '\n    '.join(doc["test_list"][0:1])
            
            code = doc['code']
           
            code= re.sub(r'[\r\n]+', '\n', code) 
             
            if '\n    ' not in code:
                if '\n\t' in code:
                    code = code.replace('\n\t','\n    ')
                elif '\n  ' in code:
                    code = code.replace('\n  ','\n    ')
                elif '\n ' in code:
                    code = code.replace('\n ','\n    ')
            code = self.prune_code(code)
            _,func_name_idx = self.getFuncName(code)
            if self.is_requirement_before == False:
                demonstration.append(f'{code[:func_name_idx]}    """\n    {description}\n    {test_example}\n    """\n{self.comment}{code[func_name_idx:]}\n')
            else:
                demonstration.append(f'"""\n    {description}\n    {test_example}\n    """\n{code[:func_name_idx]}{self.comment}{code[func_name_idx:]}\n')
            
        self.demonstration = demonstration
        return dataset
    def prune_code(self,input_code):
        code_split_list = input_code.split('def ')
        if len(code_split_list)>=2:
            raw_code = code_split_list[0]+'def '+code_split_list[-1]
        else:
            raw_code = input_code
        return raw_code

    def getFuncName(self,raw_code):
        
        
        # print([raw_code])
        start_index = raw_code.find('def')
        # extract_code = raw_code[start_index:]
        # print(extract_code)
        # end_index = extract_code.find(":")
        # start_index = start_index +end_index
        # original_end_index = start_index+4
        extract_code = raw_code[start_index:]
        original_end_index = extract_code.find('\n')+start_index+1
        func_line = raw_code[:original_end_index]
        # original_end_index = start_index+end_index+1
       
            

      
        return func_line,original_end_index
    def align_code(self,code):
        code= re.sub(r'[\r\n]+', '\n', code) 
                
        if '\n    ' not in code:
            if '\n\t' in code:
                code = code.replace('\n\t','\n    ')
            elif '\n  ' in code:
                code = code.replace('\n  ','\n    ')
            elif '\n ' in code:
                code = code.replace('\n ','\n    ')
        return code

    def get_prompt(self, doc,n_shot=None,start_retrieval_aug=False):
        """Builds the prompt for the LM to generate from.
        MBPP prompt is built following to InCoder (Fried et al.) approach
        prompt = docstring that includes one test
        """
        code = doc['code']
        code= re.sub(r'[\r\n]+', '\n', code) 
        code = self.prune_code(code)
        # print(code)
        # input()
        doc_idx = doc['task_id']
        description = doc["text"]
        test_example = doc["test_list"][0]
        # code_func_name = test_example[test_example.find('assert ')+len('assert '):test_example.find('(')]
        func_name,func_idx =self.getFuncName( code)
        self.current_func_name = func_name
        
        
        if n_shot ==None:
            
            # prompt = f'"""\n{description}\n{test_example}\n"""\n'
            if self.is_requirement_before == False:
                return f'{func_name}    """\n    {description}\n    {test_example}\n    """\n{self.comment}'
            else:
                return f'"""\n{description}\n{test_example}\n"""\n{func_name}{self.comment}'
        elif n_shot >0:
            if start_retrieval_aug ==False:
                if self.is_requirement_before == False:
                    prompt = ''.join(self.demonstration[:n_shot])+f'{func_name}    """\n    {description}\n    {test_example}\n    """\n{self.comment}'
                    self.n_shot = n_shot
                else:
                    prompt = ''.join(self.demonstration[:n_shot])+f'"""\n    {description}\n    {test_example}\n    """{func_name}\n{self.comment}'
                    self.n_shot = n_shot
            else:
                with open('/home/azureuser/myCode/retireval_aug/dataset/mbpp/ans.json','r')as f:
                    ans_list = json.load(f)
                aug_demonstration = []
                for ans in ans_list[str(doc_idx)]:
                    aug_code_description = ans["text"]
                    aug_code = ans['code']
                    aug_code = self.prune_code(aug_code)
                    aug_code_test_example = '\n    '.join(ans["test_list"][0:1])
                    
                    aug_code = self.align_code(aug_code)

                    _,aug_func_name_idx = self.getFuncName(aug_code)
                    
                    aug_demonstration.append(f'{aug_code[:aug_func_name_idx]}    """\n    {aug_code_description}\n    {aug_code_test_example}\n    """\n{self.comment}{aug_code[aug_func_name_idx:]}\n')
            
                
                prompt =  ''.join(aug_demonstration[:n_shot])+f'{func_name}    """\n    {description}\n    {test_example}\n    """\n{self.comment}'
                self.n_shot = n_shot
                self.start_retrieval_aug = True
                 # print(prompt)
        # input()
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "\n".join(doc["test_list"])

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
        
        
        if self.n_shot!=None:
            prompt = self.get_prompt(self.dataset["test"][idx],n_shot =  self.n_shot,start_retrieval_aug=self.start_retrieval_aug)

            generation = generation[len(prompt) :]
            return  self.current_func_name + self._stop_at_stop_token(generation, self.stop_words)
        prompt = self.get_prompt(self.dataset["test"][idx])
        generation = generation[len(prompt) :]

        return prompt + self._stop_at_stop_token(generation, self.stop_words)

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
