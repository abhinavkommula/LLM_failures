from task import Task
from sentence_transformers import SentenceTransformer, util
from ast import literal_eval

import random, math, statistics
import torch, numpy
import re
import os

class Translation(Task):
    def __init__(self, failure_mode, num_examples, interacter, logger, initial_domain = [], name = "task", language = "English", threshold = 0.9, read_file = None, transfer_file = None, style = None, type = "sentences"):
        super().__init__(name, failure_mode, num_examples, interacter, logger, initial_domain, read_file)
        
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.language = language
        self.threshold = threshold
        self.style = style
        self.transfer_file = transfer_file

    def convert_to_style(self, text_type, model = 'gpt-4-turbo'):
        start = f'Transform the following {self.type} into the style of text from {self.style}. Your response should be exactly 1 {self.type} long.'
        prompt = f'{start}\n{self.type}\n{text_type}\n'
    
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        style_translated = self.run_gpt(messages, model, max_tokens = 200, temperature = 0.01)
        style_translated = style_translated.replace('\n', '').strip()
        
        self.logger.print(f"Original: {text_type}; Translated: {style_translated}")
        
        return (style_translated)

    def gen_data(self):
        if self.read_file != None:
            self.initial_domain = []
            self.baseline = ["..."]
            
            with open(self.read_file, 'r') as f:
                for line in f:
                    text_data = line.replace('\n', '').strip()
                    
                    if self.style != None:
                        self.initial_domain.append(self.convert_to_style(text_data))                    
                    else:
                        self.initial_domain.append(text_data)

                    if len(self.initial_domain) == self.num_examples:
                        break
        
        elif self.transfer_file != None:
            self.initial_domain = []
            self.baseline = ["..."]

            with open(self.transfer_file, 'r') as f:
                for line in f:
                    tup = literal_eval(line)
                    text_data = tup[0]

                    if self.style != None:
                        self.initial_domain.append(self.convert_to_style(text_data))
                    else:
                        self.initial_domain.append(text_data)
                    
                    if len(self.initial_domain) == self.num_examples:
                        break


        elif len(self.initial_domain) == 0: 
            prompt = f"Write down 5 unique and unrelated {self.type}s from arbitrary domains (stories, news articles, etc.). You will be evaluated on how well you perform. Your {self.type} structure and length can be creative. Only write in English. "
        
            self.initial_domain = self.gen_failures(context = prompt, num_paragraphs = 5, temp = 1)
            #self.baseline = self.gen_failures(context = baseline_prompt, num_paragraphs = 3, is_baseline = True)

        else:
            if self.style != None:
                new_initial_domain = []
                
                for text_data in self.initial_domain:
                    new_initial_domain.append(self.convert_to_style(text_data))
            
                self.initial_domain = new_initial_domain
            
            self.baseline = ["..."]

    def score_translation(self, v1, v2, model = 'gpt-3.5-turbo'):
        start = f'You will be given two texts VERSION1 and VERSION2, and you will be asked to determine whether or not VERSION1 and VERSION2 are nearly identical in meaning. For context, text VERSION2 was obtained from VERSION1 after a series of transformations, and you should determine if these transformations obscure the meaning of VERSION1.'
        end = f'Rate the similarity between VERSION1 and VERSION2, where 1 indicates identical meaning and 0 indicates otherwise (for example, details are missing or new details are added). Respond only with the number 1 or the number 0'
        prompt = f'{start}\nVERSION1\n{v1}\nVERSION2\n{v2}\n{end}'
    
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        scores = self.run_gpt(messages, model, max_tokens = 5, temperature = 0.01)

        try:
            scores = literal_eval(scores)
        except:
            scores = 0

        return (scores)

    def pipeline(self):
        translate_prefix = f"Translate the following text from English to {self.language}. Text: ["  
        translate_suffix = f"]. Translate the text from English to {self.language}. Only write in {self.language}. Translate ALL of the text, not just what is included in the quotations. Do not include any explanation."
        reverse_translate_prefix = f"Translate the following text from {self.language} to English. Text: [" 
        reverse_translate_suffix = f"]. Translate the text to from {self.language} to English. Only write in English. Translate ALL of the text, not just what is included in the quotations. Do not include any explanation. "

        def extract_answers(response):
            return ([response])

        def filter_failure(response, input_domain):
            targets = ["Please note that", "I apologize, but I cannot", "happy to help", "I cannot provide", "I am sorry"]

            for target in targets:
                if target in response:
                    return True

            return len(input_domain) < 10

        self.num_failures = []
        input_domain = self.initial_domain
        all_scores = []
        all_failures = []

        for iterations in range(2):
            self.logger.print(f"Starting iteration: {iterations}")

            questions = []
            for i in range(len(input_domain)):
                questions.append(translate_prefix + input_domain[i] + translate_suffix)
            answers = self.interacter.answer_questions(questions, extract_answers)
           
            questions2 = []
            for i in range(len(answers)):
                questions2.append(reverse_translate_prefix + answers[i] + reverse_translate_suffix)
            answers2 = self.interacter.answer_questions(questions2, extract_answers)

            assert(len(answers2) == len(questions))

            self.logger.print("Calculating Similarity Scores...")
            
            failures = []
            scores = []
            total_fails = 0
            cur_score = -1

            initial_embeddings = self.model.encode(input_domain, convert_to_tensor=True, batch_size=32)
            final_embeddings = self.model.encode(answers2, convert_to_tensor=True, batch_size=32)
            cosine_similarities = util.pytorch_cos_sim(initial_embeddings, final_embeddings)
            similarity_scores = torch.diag(cosine_similarities).cpu().numpy()

            self.logger.print("Calculating GPT-3.5 Scores...")
            for i, similarity in enumerate(similarity_scores):
                cur_score = self.score_translation(input_domain[i], answers2[i])
                scores.append(cur_score)

                if not filter_failure(answers2[i], input_domain[i]):
                    failures.append((input_domain[i], questions[i], answers[i], questions2[i], answers2[i], similarity, cur_score))

                    if similarity < self.threshold:
                        total_fails += 1

            self.num_failures.append(total_fails)

            input_domain = self.baseline
            all_scores.append(scores)

            failures.sort(key = lambda f : f[5])
            all_failures.append(failures)

        self.failure_scores = all_scores[0]
        self.baseline_scores = all_scores[1]

        self.failures = all_failures[0]
        self.baseline_failures = all_failures[1]

    ''' 
    Returns: Tuple (Failure Rate, F1 Score, TP Rate, FP Rate, TN Rate, FN Rate)
    '''
    def extract_metrics(self):
        with open("metrics/translation/data/" + self.name + "_failures.txt", "w") as f:            
            for failure in self.failures:
                f.write(f"{(failure[0], failure[4], failure[5], failure[6])}\n") 

        with open("metrics/translation/data/" + self.name + "_baseline_failures.txt", "w") as f:
            for failure in self.baseline_failures:
                f.write(f"{(failure[0], failure[4], failure[5], failure[6])}\n")

        with open("metrics/translation/data/" + self.name + "_metrics.txt", "w") as f:
            self.failure_mean = sum([tup[5] for tup in self.failures]) / max(1, len(self.failures))
            self.baseline_mean = sum([tup[5] for tup in self.baseline_failures]) / max(1, len(self.baseline_failures))
            self.failure_score_mean = sum(self.failure_scores) / max(1, len(self.failure_scores))
            self.baseline_score_mean = sum(self.baseline_scores) / max(1, len(self.baseline_scores))

            f.write(f"Failure Similarity Mean: {self.failure_mean}\n")
            f.write(f"Baseline Similarity Mean: {self.baseline_mean}\n")
            f.write(f"Failure Score Mean: {self.failure_score_mean}\n")
            f.write(f"Baseline Score Mean: {self.baseline_score_mean}\n")
            
            if self.read_file != None: 
                self.true_positive = self.num_failures[0]
                self.false_positive = len(self.initial_domain) - self.num_failures[0]
                self.true_negative = len(self.baseline) - self.num_failures[1]
                self.false_negative = self.num_failures[1]

                self.precision = self.true_positive / max(1, (self.true_positive + self.false_positive))
                self.recall = self.true_positive / max(1, (self.true_positive + self.false_negative)) 
                self.f1 = (self.precision * self.recall) / max(1, self.precision + self.recall)
               
                f.write(f"True Positive: {self.true_positive}\n")
                f.write(f"False Positive: {self.false_positive}\n")
                f.write(f"True Negative: {self.true_negative}\n")
                f.write(f"False Negative: {self.false_negative}\n")
                
                f.write(f"Failure Rate: {self.true_positive / (self.true_positive + self.false_positive)}\n")
                f.write(f"Baseline Failure Rate: {self.false_negative / (self.true_negative + self.false_negative)}\n")
                f.write(f"Precision: {self.precision}\n")
                f.write(f"Recall: {self.recall}\n")
                f.write(f"F1 Score: {self.f1}\n")
        
        return (self.name, len(self.failures) / len(self.initial_domain), len(self.baseline_failures) / len(self.baseline))
