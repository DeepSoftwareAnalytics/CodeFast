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
import json
from evaluate import load

from lm_eval.base import Task
import pickle
_CITATION = """
@article{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
}
"""


class MBPP_AUG(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "mbpp"

    def __init__(self,prompt=None,mode=None):
        super().__init__(
            # stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/","def",'import','from'],
            stop_words= ['def','#test','#Test','from','import','# if','# Input','# main()','# '],
            requires_execution=True,
        )
        self.prompt = prompt
        self.mode = mode
        if self.prompt == 'self_planning':
            with open('prompt/icl_examples.json','r')as f:
                self.icl_examples = json.load(f)

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        # the wrong split of mbpp can be loaded with old datasets cache
        assert (
            len(dataset) == 500
        ), "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        if self.mode == "test":
            print(type(dataset))
            dataset = dataset.select(list(range(10)))
            return dataset
        else:
            return dataset

    def getFuncName(self,raw_code):
        start_index = raw_code.find('def')
        extract_code = raw_code[start_index:]
        end_index = extract_code.find(":")
        original_end_index = start_index+end_index+3 
        return raw_code[:original_end_index],original_end_index

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        MBPP prompt is built following to InCoder (Fried et al.) approach
        prompt = docstring that includes one test
        """
        
        description = doc["text"]
        test_example = doc["test_list"][0]
        doc_idx = doc['task_id']
        func_name,_ = self.getFuncName(doc['code'])
        func_name = func_name.replace('\r','')+'\n'
        # print(func_name)
        # input()
        if 'acecoder' == self.prompt:
            with open('/home/azureuser/myCode/retireval_aug/dataset/mbpp/prompt.json','r')as f:
                ans_list = json.load(f)
            prompt = ''
            for ans in ans_list[:3]:
                
                ans_description = ans["text"]
                ans_code = ans['code']
                # ans_test = '\n    '.join(['#'+ans_test_case for ans_test_case in ans["test_list"]])
                _,end_idx = self.getFuncName(ans_code)
                #prompt += f'{ans_code[:end_idx]}    #{ans_description}\n    #Examples\n    {ans_test}\n{ans_code[end_idx:]}\n'
                prompt += f'{ans_code[:end_idx]}    #{ans_description}\n{ans_code[end_idx:]}\n'
            
            prompt += f'{func_name}    #{description}\n'

          
        elif self.prompt == 'acecoder_aug':
            with open('/home/azureuser/myCode/retireval_aug/dataset/mbpp/ans.json','r')as f:
                ans_list = json.load(f)
            prompt = ''
            for ans in ans_list[str(doc_idx)]:
                ans_description = ans["text"]
                ans_code = ans['code']
                ans_test = '\n    '.join(['#'+ans_test_case for ans_test_case in ans["test_list"]])
                _,end_idx = self.getFuncName(ans_code)
                prompt += f'{ans_code[:end_idx]}    #{ans_description}\n    #Examples\n    {ans_test}\n{ans_code[end_idx:]}\n'
            
            prompt += f'{func_name}    #{description}\n    #Examples\n    #assert '
        elif self.prompt == 'acecoder_aug_test_case':
            with open('/home/azureuser/myCode/retireval_aug/dataset/mbpp/ans.json','r')as f:
                ans_list = json.load(f)
            prompt = ''
            for ans in ans_list[str(doc_idx)]:
                ans_description = ans["text"]
                ans_test = '\n'.join(['#'+ans_test_case for ans_test_case in ans["test_list"]])
                ans_code = ans['code']
                prompt += f'[requirement]\n#{ans_description}\n[source code]\n{ans_test}\n{ans_code}\n'
            prompt += f'[requirement]\n#{description}\n[source code]\n'
        elif self.prompt == 'acecoder_aug_no_test_case':
            with open('/home/azureuser/myCode/retireval_aug/dataset/mbpp/ans.json','r')as f:
                ans_list = json.load(f)
            prompt = ''
            for ans in ans_list[str(doc_idx)]:
                ans_description = ans["text"]
                ans_code = ans['code']
                # ans_test = '\n    '.join(['#'+ans_test_case for ans_test_case in ans["test_list"]])
                _,end_idx = self.getFuncName(ans_code)
                #prompt += f'{ans_code[:end_idx]}    #{ans_description}\n    #Examples\n    {ans_test}\n{ans_code[end_idx:]}\n'
                prompt += f'{ans_code[:end_idx]}    #{ans_description}\n{ans_code[end_idx:]}\n'
            
            prompt += f'{func_name}    #{description}\n'
        elif self.prompt == 'self_planning':
            planning_examples = self.icl_examples[self.prompt]['mbpp']
            prompt = planning_examples+'\n<planning>'
            # test_example = doc["test_list"][0].replace('#assert ','').replace('==',' returns ')
            prompt += f"{description}\nLet's think step by step.\n"
            return prompt+'<func_name>'+func_name

        else:
            
            prompt = f'"""\n{description}\n{test_example}\n"""\n'
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
        prompt = self.get_prompt(self.dataset["test"][idx])
        func_name,_ = self.getFuncName(self.dataset["test"][idx]['code'])
        generation = generation[len(prompt) :]
        if self.mode == 'test':
            print("original prompt:\n")
            print(prompt+'\n\n')
            print("original generation:\n")
            print(generation)
            print("after extract\n")
            print( self._stop_at_stop_token(generation, self.stop_words)+'\n')
            return prompt +'\n'+ self._stop_at_stop_token(generation, self.stop_words)
        else:
            return func_name+'\n'+self._stop_at_stop_token(generation, self.stop_words)
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_metric = load("code_eval")
        results, test_case_units = code_metric.compute(
            references=references,
            predictions=generations,
            k=[1,3,5]
        )
        
        return results,test_case_units

