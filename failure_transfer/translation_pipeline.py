from task import Task
from sentence_transformers import SentenceTransformer, util
from ast import literal_eval

import random, math, statistics
import re
import os

class Translation(Task):
    def __init__(self, failure_mode, num_examples, interacter, initial_domain = [], name = "task", language = "English", threshold = 0.9, read_file = None):
        super().__init__(name, failure_mode, num_examples, interacter, initial_domain, read_file)
        
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.language = language
        self.threshold = threshold

    def gen_data(self):
        if self.read_file != None:
            paragraphs_list = []

            with open(self.read_file, 'r') as f:
                for line in f:
                    tup = literal_eval(line)
                    paragraphs_list.append((tup[0], tup[2]))
            
            idx = int(len(paragraphs_list) * 0.1)
            while idx < len(paragraphs_list) and paragraphs_list[idx][1] < 0.9:
                idx += 1

            self.initial_domain = [el[0] for el in paragraphs_list[:idx]]
            print("Mean failure transfer:", statistics.mean([el[1] for el in paragraphs_list[:idx]]))

            random.shuffle(paragraphs_list)
            self.baseline = [el[0] for el in paragraphs_list[:1000]]
            
            print("Mean baseline:", statistics.mean([el[1] for el in paragraphs_list[:1000]]))

        elif len(self.initial_domain) == 0: 
            prompt = "Write down 5 unique and unrelated sentences from arbitrary domains (stories, news articles, etc.). You will be evaluated on how well you perform. Your sentence structure and length can be creative. Only write in English. "
        
            self.initial_domain = self.gen_failures(context = prompt, num_paragraphs = 5, temp = 1)
            #self.baseline = self.gen_failures(context = baseline_prompt, num_paragraphs = 3, is_baseline = True)

        else:
            self.baseline = ["..."]

    def score_translation(self, v1, v2, model = 'gpt-3.5-turbo'):
        start = f'You will be given two texts VERSION1 and VERSION2, and you will be asked to determine whether or not VERSION1 and VERSION2 are nearly identical. For context, text VERSION2 was obtained from VERSION1 after a series of transformations, and you should determine if these transformations obscure the meaning of VERSION1.'
        end = f'Rate the similarity between VERSION1 and VERSION2, where 1 indicates identical meaning and 0 indicates otherwise (for example, details are missing or incorrect). Respond only with the number 1 or the number 0'
        prompt = f'{start}\nVERSION1\n{v1}\nVERSION2\n{v2}\n{end}'
    
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        scores = self.run_gpt(messages, model, max_tokens = 1500, temperature = 0.01)

        try:
            scores = literal_eval(scores)
        except:
            scores = 0

        return (scores)

    def pipeline(self):
        translate_prefix = f"Translate the following text from English to {self.language}: ["  
        translate_suffix = f"]. Translate the text to {self.language} to the best of your ability. Only write your translation. Only write in {self.language}. "
        reverse_translate_prefix = f"Translate the following text from {self.language} to English: [" 
        reverse_translate_suffix = f"]. Translate the text to English to the best of your ability. Only write your translation. Only write in English. "

        def extract_answers(response):
            if ':' not in response:
                return ([response])

            return ([':'.join(response.split(":")[1:])])

        def filter_failure(response, input_domain):
            targets = ["Please note that", "I apologize, but I cannot", "happy to help"]

            for target in targets:
                if target in response:
                    return True

            return len(input_domain) < 10

        input_domain = self.initial_domain
        all_scores = []
        all_failures = []

        for iterations in range(2):
            questions = []
            for i in range(len(input_domain)):
                questions.append(translate_prefix + input_domain[i] + translate_suffix)
            answers = self.interacter.answer_questions(questions, extract_answers, 30)
            
            questions2 = []
            for i in range(len(answers)):
                questions2.append(reverse_translate_prefix + answers[i] + reverse_translate_suffix)
            answers2 = self.interacter.answer_questions(questions2, extract_answers, 30)

            assert(len(answers2) == len(questions))

            failures = []
            scores = []

            for i in range(len(answers2)):
                initial_embedding = self.model.encode(input_domain[i])
                final_embedding = self.model.encode(answers2[i])
                similarity = util.pytorch_cos_sim(initial_embedding, final_embedding).item()
                
                cur_score = self.score_translation(input_domain[i], answers2[i])
                scores.append(cur_score)

                if similarity < self.threshold and not filter_failure(answers2[i], input_domain[i]):
                    if len(answers[i]) > 1 and len(answers2[i]) > 1:
                        failures.append((input_domain[i], questions[i], answers[i], questions2[i], answers2[i], similarity, cur_score))

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

        return (self.name, len(self.failures) / len(self.initial_domain), len(self.baseline_failures) / len(self.baseline))
