from task import Task
from sentence_transformers import SentenceTransformer, util

import random, math
import re
import os

class Translation(Task):
    def __init__(self, failure_mode, num_examples, language, threshold = 0.9):
        super().__init__("translation_task", failure_mode, num_examples)

        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.language = language
        self.threshold = threshold

    def pipeline(self):
        translate_prefix = f"Translate the following text from English to {self.language}. "  
        reverse_translate_prefix = f"Translate the following text from {self.language} to English. " 

        def extract_answers(response):
            return ([response])

        input_domain = self.initial_domain
        all_failures = []
        print("Scraping Initial Domain...")

        for iterations in range(2):
            questions = []
            for i in range(len(input_domain)):
                questions.append(translate_prefix + input_domain)
            answers = self.interacter.answer_questions(questions, extract_answers)

            questions2 = []
            for i in range(len(answers)):
                questions2.append(reverse_translate_prefix + answers[i])
            answers2 = self.interacter.answer_questions(questions2, extract_answers)

            failures = []
            for i in range(len(answers2)):
                print(f"Comparing initial text: {input_domain}\n with final text: {answers2[i]}\n")
            
                initial_embedding = self.model.encode(input_domain)
                final_embedding = self.model.encode(answers2[i])
                similarity = util.pytorch_cos_sim(initial_embedding, final_embedding)
    
                print(f"Semantic Similarity Score: {similarity}\n")
            
                if similarity < self.threshold:
                    failures.append((input_domain, answer2[i]))

            print("Scraping Baseline...")
            input_domain = self.baseline
            all_failures.append(failures)

        self.failures = all_failures[0]
        self.baseline_failures = all_failures[1]
