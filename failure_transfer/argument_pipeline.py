import numpy as np
import random, math, statistics
import spacy, torch
import re
import os

from task import Task
from ast import literal_eval

class Argument(Task):
    def __init__(self, failure_mode, num_examples, interacter, logger, initial_domain = [], name = "argument_task", read_file = None):
        super().__init__(name, failure_mode, num_examples, interacter, logger, initial_domain, read_file)

    def gen_data(self):
        self.baseline = [] 

    def score_argument(self, original, argument,  model = 'gpt-4-turbo'):
        start = f'You will be given original text ORIGINAL and a main argument ARGUMENT, and you will be asked to score the quality of the argument extracted ARGUMENT on a scale from 0 to 10. A strong argument should capture the central idea of the original text ORIGINAL. '
        end = f'Rate how strong ARGUMENT is on a scale of 0 to 10 where a 10 indicates that ARGUMENT contains the main idea and overarching argument of the text, while a 0 indicates that the argument is not related at all to the main idea of the text. Respond only with an integer from 0 to 10. '
        prompt = f'{start}\nORIGINAL:\n{original}\nARGUMENT:\n{argument}\n{end}'
    
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        scores = self.run_gpt(messages, model, max_tokens = 1000, temperature = 0)

        try:
            scores = literal_eval(scores)
        except:
            scores = 0

        return (scores)

    def pipeline(self):
        argument_prefix = "Given the following text, please extract the main argument or central idea. Text: [\n"
        argument_suffix = "]\nFor the text provided, please write the main argument or central idea in exactly one sentence. Format your sentence as 'Argument: ...', and do not include any explanation. "
        
        def extract_answers(response):
            if "Argument:" in response:
                return [response.split("Argument:")[1]]
            
            return [response]
                
        input_domain = self.initial_domain
        all_failures = []
        failures = []

        for iteration in range(1):
            questions = []
            for i in range(len(input_domain)):
                questions.append(argument_prefix + input_domain[i] + argument_suffix)
                
            answers = self.interacter.answer_questions(questions, extract_answers)
                    
            assert(len(questions) == len(answers))
                    
            for i in range(len(input_domain)):         
                failures.append((input_domain[i], answers[i], self.score_argument(input_domain[i], answers[i])))

            input_domain = self.baseline
            all_failures.append(failures)

        self.failures = all_failures[0]
        #self.baseline_failures = all_failures[1]

    def extract_metrics(self):
        with open("metrics/argument/data/" + self.name +  "_failures.txt", "w") as f:
            for failure in self.failures:
                f.write(f"{failure}\n")
                       
        return (self.name, -1, -1)
