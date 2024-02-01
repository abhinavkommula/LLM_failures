from task import Task
from sentence_transformers import SentenceTransformer, util

import random, math
import re
import os

class Translation(Task):
    def __init__(self, failure_mode, num_examples, interacter, language, threshold = 0.9):
        super().__init__("translation_task", failure_mode, num_examples, interacter)

        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.language = language
        self.threshold = threshold

    def gen_data(self):
        prompt = "Write down 5 additional paragraphs that when passed as input to a language model with the following failure mode might elicit failures in language translation. You will be evaluated on how well you perform. Your sentence structure and length can be creative; extrapolate based on the failure mode. Be both creative and cautious. "

        baseline_prompt = "Write down 5 additional paragraphs that when passed as input to a language model might elicit failures in language translation. You will be evaluated on how well you perform. Your sentence structure and length can be creative; be both creative and cautious. "

        self.initial_domain = self.gen_failures(context = prompt, num_paragraphs = 5)
        self.baseline = self.gen_failures(context = baseline_prompt, num_paragraphs = 5, is_baseline = True)

    def pipeline(self):
        translate_prefix = f"Translate the following paragraph from English to {self.language}. Only write the translated paragraph, do not include the original text. Paragraph: "  
        reverse_translate_prefix = f"Translate the following paragraph from {self.language} to English. Only write the translated paragraph, do not include the original text. Paragraph: " 

        def extract_answers(response):
            return ([response])

        input_domain = self.initial_domain
        all_failures = []

        for iterations in range(2):
            questions = []
            for i in range(len(input_domain)):
                questions.append(translate_prefix + input_domain[i])
            answers = self.interacter.answer_questions(questions, extract_answers)

            questions2 = []
            for i in range(len(answers)):
                questions2.append(reverse_translate_prefix + answers[i])
            answers2 = self.interacter.answer_questions(questions2, extract_answers)

            failures = []
            for i in range(len(answers2)):
                initial_embedding = self.model.encode(input_domain[i])
                final_embedding = self.model.encode(answers2[i])
                similarity = util.pytorch_cos_sim(initial_embedding, final_embedding).item()
    
                if similarity < self.threshold:
                    failures.append((input_domain[i], questions2[i], answers2[i], similarity))

            input_domain = self.baseline
            all_failures.append(failures)

        self.failures = all_failures[0]
        self.baseline_failures = all_failures[1]
