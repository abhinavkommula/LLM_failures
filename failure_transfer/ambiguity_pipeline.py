import numpy as np
import random, math, statistics
import spacy, torch
import re
import os

from task import Task
from ast import literal_eval

class Ambiguity(Task):
    def __init__(self, failure_mode, num_examples, interacter, logger, initial_domain = [], name = "ambiguity_task", read_file = None):
        super().__init__(name, failure_mode, num_examples, interacter, logger, initial_domain, read_file)

    def gen_data(self):
        self.baseline = [] 

    def score_ambiguity(self, v1, v2,  model = 'gpt-4-turbo'):
        start = f'You will be given two texts VERSION1 and VERSION2, and you will be asked to determine whether or not VERSION1 and VERSION2 are nearly identical in meaning. For context, text VERSION2 was obtained from VERSION1 after a series of transformations, and you should determine if these transformations obscure the meaning of VERSION1.'
        end = f'Rate the similarity between VERSION1 and VERSION2, where 1 indicates identical meaning and 0 indicates otherwise (for example, details to not mean the exact same thing). Respond only with the number 1 or the number 0'
        prompt = f'{start}\nVERSION1\n{v1}\nVERSION2\n{v2}\n{end}'
    
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        scores = self.run_gpt(messages, model, max_tokens = 1000, temperature = 0)

        try:
            scores = literal_eval(scores)
        except:
            scores = 0

        return (scores)

    def pipeline(self):
        ambiguity_prefix = "Given the following text, rewrite all sentences so that any ambiguous phrases or ideas that could have multiple interpretations are more specific within the context of the text. Text: [\n"
        ambiguity_suffix = "]\nFor each sentence in the original text, rewrite all ambiguous phrases or ideas so that they are more specific, and mean the same thing within the context of the text. Format your text as 'Text: ...', and do not include any explanation. "
        
        def extract_answers(response):
            if "Text:" in response:
                return [response.split("Text:")[1]]
            
            return [response]
                
        input_domain = self.initial_domain
        all_failures = []
        failures = []

        for iteration in range(1):
            questions = []
            for i in range(len(input_domain)):
                questions.append(ambiguity_prefix + input_domain[i] + ambiguity_suffix)
                
            answers = self.interacter.answer_questions(questions, extract_answers)
                    
            assert(len(questions) == len(answers))
                    
            for i in range(len(input_domain)):         
                failures.append((input_domain[i], answers[i], self.score_ambiguity(input_domain[i], answers[i])))

            input_domain = self.baseline
            all_failures.append(failures)

        self.failures = all_failures[0]
        #self.baseline_failures = all_failures[1]

    def extract_metrics(self):
        with open("metrics/ambiguity/data/" + self.name +  "_failures.txt", "w") as f:
            for failure in self.failures:
                f.write(f"{failure}\n")
                       
        return (self.name, -1, -1)
