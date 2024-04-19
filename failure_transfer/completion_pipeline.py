import numpy as np
import random, math, statistics
import spacy, torch
import re
import os

from task import Task
from ast import literal_eval

class Completion(Task):
    def __init__(self, failure_mode, num_examples, interacter, logger, initial_domain = [], name = "completion_task", read_file = None):
        super().__init__(name, failure_mode, num_examples, interacter, logger, initial_domain, read_file)
        
    def gen_data(self):
        self.baseline = [] 

    def score_entailment(self, original, entailment, model = 'gpt-4-turbo'):
        start = f'You will be given original text ORIGINAL and a completed paragraph COMPLETED, and you will be asked to score the quality of COMPLETED. A completion should logically follow from the original text ORIGINAL and resolve any incomplete ideas or implications in ORIGINAL.'
        end = f'Rate how strong COMPLETED is on a scale of 0 to 1 where a 1 indicates that COMPLETED logically follows from text ORIGINAL and completes all ideas presented in the text in a natural manner, while a 0 indicates that the completion makes little logical sense given the original paragraph or does not naturally complete ideas presented in original. Respond only with either a 1 or a 0.'
        prompt = f'{start}\nORIGINAL:\n{original}\COMPLETION:\n{entailment}\n{end}'
    
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        scores = self.run_gpt(messages, model, max_tokens = 1000, temperature = 0)

        try:
            scores = literal_eval(scores)
        except:
            scores = 0

        return (scores)

    def pipeline(self):
        entailment_prefix = "Given the following text, please write a paragraph that completes it. Text: [\n"
        entailment_suffix = "]\nFor the text provided, write a subsequent paragraph that naturally follows and completes all ideas presented naturally. Do not include any explanation. "

        def extract_answers(response):
            return [response]
                
        input_domain = self.initial_domain
        all_failures = []
        failures = []

        for iteration in range(1):
            questions = []
            for i in range(len(input_domain)):
                questions.append(entailment_prefix + input_domain[i] + entailment_suffix)
                
            answers = self.interacter.answer_questions(questions, extract_answers)
                    
            assert(len(questions) == len(answers))
                
            for i in range(len(input_domain)):    
                failures.append((input_domain[i], answers[i], self.score_entailment(input_domain[i], answers[i])))

            input_domain = self.baseline
            all_failures.append(failures)

        self.failures = all_failures[0]
        #self.baseline_failures = all_failures[1]

    def extract_metrics(self):
        with open("metrics/completion/data/" + self.name +  "_failures.txt", "w") as f:
            for failure in self.failures:
                f.write(f"{failure}\n")
                       
        return (self.name, -1, -1)
