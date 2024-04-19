import numpy as np
import random, math, statistics
import spacy
import re
import os

from task import Task
from ast import literal_eval
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge

class StyleGeneration(Task):
    def __init__(self, failure_mode, num_examples, interacter, logger, initial_domain = [], name = "score_task", style = "english", read_file = None):
        super().__init__(name, failure_mode, num_examples, interacter, logger, initial_domain, read_file)
        
        self.style = style

    def gen_style(self, prompt, model = 'gpt-4-turbo'):
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        output_style = self.run_gpt(messages, model, max_tokens = 1000, temperature = 0)
        output_style = output_style.replace('\n', '')
       
        return (output_style)        

    def gen_data(self):
        if len(self.initial_domain) == 0:
            self.logger.print("Num examples: " + str(self.num_examples))
            self.initial_domain = self.scraper.get_examples(self.num_examples)
            self.logger.print(str(len(self.initial_domain)))
            self.baseline = ["..."]
             
        else:
            self.baseline = ["..."] 
        
    def evaluate_style(self, original, stylistic,  model = 'gpt-4-turbo'):
        start = f'I will provide you with two texts, the original text ORIGINAL and the generated text STYLISTIC which was created in the style of {self.style}.'
        end = f'Evaluate whether STYLISTIC is text adheres to the style of {self.style} generated from ORIGINAL, where 1 indicates an accurate stylistic representation and 0 indicates 0 otherwise. Respond only with either a 0 or a 1.'
        prompt = f'{start}\nORIGINAL:\n{original}\nSTYLISTIC:\n{stylistic}\n{end}'
    
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        scores = self.run_gpt(messages, model, max_tokens = 1000, temperature = 0)

        try:
            scores = literal_eval(scores)
        except:
            scores = 0

        return (scores)

    def pipeline(self):
        style_prefix = f"Rewrite the following text in the style of {self.style}. Text: [\n"
        style_suffix = f"\n]. Rewrite all of the provided text in the style of {self.style} to the best of your ability. Format your output as: 'Style: ...', and do not include any explanation or additional dialogue."

        def extract_answers(response):
            if ':' not in response:
                return ([response])
            
            return ([':'.join(response.split(":")[1:])])

        input_domain = self.initial_domain
        all_scores = []
        all_failures = []

        for iterations in range(2):
            questions = []
            for i in range(len(input_domain)):
                questions.append(style_prefix + input_domain[i] + style_suffix)
            answers = self.interacter.answer_questions(questions, extract_answers)
            
            assert(len(questions) == len(answers))

            failures = []
            scores = []

            for i in range(len(questions)):
                gpt_eval = self.evaluate_style(input_domain[i], answers[i])
                failures.append((input_domain[i], answers[i], gpt_eval))

            input_domain = self.baseline
            all_scores.append(scores)

            all_failures.append(failures)

        self.failure_scores = all_scores[0]
        self.baseline_scores = all_scores[1]

        self.failures = all_failures[0]
        self.baseline_failures = all_failures[1]

    def extract_metrics(self):
        with open("metrics/style_gen/data/" + self.name +  "_failures.txt", "w") as f:
            for failure in self.failures:
                f.write(f"{failure}\n")

        return (self.name, -1, -1)
