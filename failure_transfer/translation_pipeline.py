from task import Task
from sentence_transformers import SentenceTransformer, util
from ast import literal_eval

import random, math, statistics
import re
import os

class Translation(Task):
    def __init__(self, failure_mode, num_examples, interacter, name = "task", language = "English", threshold = 0.9, read_file = None):
        super().__init__(name, failure_mode, num_examples, interacter, read_file)
        
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.language = language
        self.threshold = threshold

    def gen_data(self):
        if self.read_file != None:
            paragraphs_list = []

            with open(self.read_file, 'r') as f:
                for line in f:
                    tup = literal_eval(line)
                    paragraphs_list.append((tup[0], tup[5]))
            
            idx = 5
            while paragraphs_list[idx][1] < 0.9:
                idx += 1

            self.initial_domain = [el[0] for el in paragraphs_list[:idx]]
            print("Mean failure transfer:", statistics.mean([el[1] for el in paragraphs_list[:idx]]))

            random.shuffle(paragraphs_list)
            self.baseline = [el[0] for el in paragraphs_list[:100]]
            
            print("Mean baseline:", statistics.mean([el[1] for el in paragraphs_list[:100]]))

            return

        prompt = "Write down 3 separate paragraphs from arbitrary domains (short stories, news articles, etc.). You will be evaluated on how well you perform. Your sentence structure and length can be creative. Only write in English. "

        baseline_prompt = "Write down 3 separate paragraphs from arbitrary domains (short domains, news articles, etc.). You will be evaluated on how well you perform. Your sentence structure and length can be creative, extrapolate if necessary. Only write in English. "
        
        self.initial_domain = self.gen_failures(context = prompt, num_paragraphs = 3)
        self.baseline = ["..."]
        #self.baseline = self.gen_failures(context = baseline_prompt, num_paragraphs = 3, is_baseline = True)

    def pipeline(self):
        translate_prefix = f"Translate the following text from English to {self.language}: ["  
        translate_suffix = f"]. Translate the text to {self.language} to the best of your ability. Only write your translation, do not include the original text. Only write in {self.language}. "
        reverse_translate_prefix = f"Translate the following text from {self.language} to English: [" 
        reverse_translate_suffix = f"]. Translate the text to English to the best of your ability. Only write your translation, do not include the original text. Only write in English. "

        def extract_answers(response):
            if ':' not in response:
                return ([response])

            return ([':'.join(response.split(":")[1:])])

        input_domain = self.initial_domain
        all_failures = []

        for iterations in range(2):
            questions = []
            for i in range(len(input_domain)):
                questions.append(translate_prefix + input_domain[i] + translate_suffix)
            answers = self.interacter.answer_questions(questions, extract_answers)
            
            questions2 = []
            for i in range(len(answers)):
                questions2.append(reverse_translate_prefix + answers[i] + reverse_translate_suffix)
            answers2 = self.interacter.answer_questions(questions2, extract_answers)

            assert(len(answers2) == len(questions))

            failures = []
            for i in range(len(answers2)):
                initial_embedding = self.model.encode(input_domain[i])
                final_embedding = self.model.encode(answers2[i])
                similarity = util.pytorch_cos_sim(initial_embedding, final_embedding).item()

                if similarity < self.threshold and answers2[i] != '':
                    failures.append((input_domain[i], questions[i], answers[i], questions2[i], answers2[i], similarity))

            input_domain = self.baseline
            failures.sort(key = lambda f : f[5])
            all_failures.append(failures)

        self.failures = all_failures[0]
        self.baseline_failures = all_failures[1]

    ''' 
    Returns: Tuple (Failure Rate, F1 Score, TP Rate, FP Rate, TN Rate, FN Rate)
    '''
    def extract_metrics(self):
        with open("metrics/translation/" + self.name + "_failures.txt", "w") as f:
            for failure in self.failures:
                f.write(f"{failure}\n")

        with open("metrics/translation/" + self.name + "_baseline_failures.txt", "w") as f:
            for failure in self.baseline_failures:
                f.write(f"{failure}\n")

        with open("metrics/translation/" + self.name + "_metrics.txt", "w") as f:
            self.failure_mean = sum([tup[5] for tup in self.failures]) / max(1, len(self.failures))
            self.baseline_mean = sum([tup[5] for tup in self.baseline_failures]) / max(1, len(self.baseline_failures))

            f.write(f"Failure Similarity Mean: {self.failure_mean}\n")
            f.write(f"Baseline Similarity Mean: {self.baseline_mean}\n")

        return (self.name, len(self.failures) / len(self.initial_domain), len(self.baseline_failures) / len(self.baseline))
