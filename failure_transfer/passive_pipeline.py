import numpy as np
import random, math, statistics
import spacy, torch
import re
import os

from task import Task
from ast import literal_eval

class Passive(Task):
    def __init__(self, failure_mode, num_examples, interacter, logger, initial_domain = [], name = "passive_task", read_file = None):
        super().__init__(name, failure_mode, num_examples, interacter, logger, initial_domain, read_file)

    def gen_data(self):
        self.baseline = [] 

    def score_passive(self, original, passive,  model = 'gpt-4-turbo'):
        start = f'You will be given original text ORIGINAL and a passive-voice version PASSIVE, and you will be asked to score the quality of PASSIVE on a scale from 0 to 1. A strong passive-voice text should have the same meaning as the original text ORIGINAL while being written in the passive-voice.'
        end = f'Rate how strong PASSIVE is on a scale of 0 to 1 where a 1 indicates that PASSIVE has identical meaning to the original text while being written in the passive voice, while a 0 indicates that either the meaning is significantly different or the text is not accurately written in the passive voic. Respond only with either the number 1 or the number 0.'
        prompt = f'{start}\nORIGINAL:\n{original}\PASSIVE:\n{passive}\n{end}'
    
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        scores = self.run_gpt(messages, model, max_tokens = 1000, temperature = 0)

        try:
            scores = literal_eval(scores)
        except:
            scores = 0

        return (scores)

    def pipeline(self):
        passive_prefix = "Given the following text, rewrite it in the passive voice. Text: [\n"
        passive_suffix = "]\nRewrite the text in the passive voice. Your output as 'Text: ...', and do not include any explanation. "
        
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
                questions.append(passive_prefix + input_domain[i] + passive_suffix)
                
            answers = self.interacter.answer_questions(questions, extract_answers)
                    
            assert(len(questions) == len(answers))
                    
            for i in range(len(input_domain)):         
                failures.append((input_domain[i], answers[i], self.score_passive(input_domain[i], answers[i])))

            input_domain = self.baseline
            all_failures.append(failures)

        self.failures = all_failures[0]
        #self.baseline_failures = all_failures[1]

    def extract_metrics(self):
        with open("metrics/passive/data/" + self.name +  "_failures.txt", "w") as f:
            for failure in self.failures:
                f.write(f"{failure}\n")
                       
        return (self.name, -1, -1)
