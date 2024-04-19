import numpy as np
import random, math, statistics
import spacy, torch
import re
import os

from task import Task
from ast import literal_eval

class SentenceSuccinct(Task):
    def __init__(self, failure_mode, num_examples, interacter, logger, initial_domain = [], name = "sentence_succinct_task", read_file = None):
        super().__init__(name, failure_mode, num_examples, interacter, logger, initial_domain, read_file)

    def gen_data(self):
        self.baseline = [] 

    def score_succinct(self, original, succinct,  model = 'gpt-4-turbo'):
        start = f'You will be given original text ORIGINAL and a more succinct version SUCCINCT, and you will be asked to score the quality of SUCCINCT on a scale from 0 to 10. A strong succinct text should have sentences that are shorter without losing the meaning of the original sentence.'
        end = f'Rate how strong SUCCINCT is on a scale of 0 to 10 where a 10 indicates that SUCCINCT has identical meaning to the original text while having shorter sentences, while a 0 indicates that either the meaning is significantly different or the sentences are not shorter. Respond only with an integer from 0 to 10.'
        prompt = f'{start}\nORIGINAL:\n{original}\nSUCCINCT:\n{succinct}\n{end}'
    
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        scores = self.run_gpt(messages, model, max_tokens = 1000, temperature = 0)

        try:
            scores = literal_eval(scores)
        except:
            scores = 0

        return (scores)

    def pipeline(self):
        succinct_prefix = "Given the following text, please modify adjectives, adverbs, or phrases within each sentence to make them as succinct as possible without losing the meaning of the original sentence.  Text: [\n"
        succinct_suffix = "]\nFor each sentence in the text provided, modify adjectives, adverbs, or phrases to make sentences as succinct as possible without losing meaning of the original sentences. Format your output as 'Text: ...', and do not include any explanation. "
        
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
                questions.append(succinct_prefix + input_domain[i] + succinct_suffix)
                
            answers = self.interacter.answer_questions(questions, extract_answers)
                    
            assert(len(questions) == len(answers))
                    
            for i in range(len(input_domain)):         
                failures.append((input_domain[i], answers[i], self.score_succinct(input_domain[i], answers[i])))

            input_domain = self.baseline
            all_failures.append(failures)

        self.failures = all_failures[0]
        #self.baseline_failures = all_failures[1]

    def extract_metrics(self):
        with open("metrics/sentence_succinct/data/" + self.name +  "_failures.txt", "w") as f:
            for failure in self.failures:
                f.write(f"{failure}\n")
                       
        return (self.name, -1, -1)
