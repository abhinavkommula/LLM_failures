from task import Task
from ast import literal_eval
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance

import numpy as np
import random, math, statistics
import re
import os

class SummarizationScore(Task):
    def __init__(self, failure_mode, num_examples, interacter, name = "score_task", read_file = None):
        super().__init__(name, failure_mode, num_examples, interacter, read_file)
        
        self.domain = domain
    
    def gen_data(self):
        if self.read_file != None:
            paragraphs_list = []

            with open(self.read_file, 'r') as f:
                for line in f:
                    tup = literal_eval(line)
                    paragraphs_list.append((tup[0], tup[2]))
                
            idx = min(len(paragraphs_list) - 1, 5)
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

    def score_summary(self, summary, model = 'gpt-3.5-turbo'):
        start = f'You will be given a summary SUMMARY, and you will be asked to score the quality of the summary on a scale from 1 to 10. A strong summary should be as succinct as possible, contain only details necessary to the story, and have logical transitions between ideas. '
        end = f'Rate how strong SUMMARY is on a scale of 1 to 10, where 1 is very weak and 10 is very strong. Respond only with a number from 1 to 10. '
        prompt = f'{start}\nSUMMARY\n{summary}\n{end}'
    
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        scores = self.run_gpt(messages, model, max_tokens = 1000, temperature = 0)

        return (literal_eval(scores))

    def pipeline(self):
        summarize_prefix = f"Please summarize the following text as succinctly as possible. Only write 2 sentences or less. "  
        
        def extract_answers(response):
            return ([response])

        input_domain = self.initial_domain
        all_scores = []
        all_failures = []

        for iterations in range(2):
            questions = []
            for i in range(len(input_domain)):
                questions.append(summarize_prefix + input_domain[i])
            answers = self.interacter.answer_questions(questions, extract_answers)
            
            failures = []
            scores = []

            for i in range(len(questions)):
                scores.append(self.score_summary(answers[i]))
    
                if scores[-1] <= 4:
                    failures.append((questions[i], answers[i], scores[-1]))

            input_domain = self.baseline
            all_scores.append(scores)
            all_failures.append(failures)

        self.failure_scores = all_scores[0]
        self.baseline_scores = all_scores[1]

        self.failures = all_failures[0]
        self.baseline_failures = all_failures[1]

    def extract_metrics(self):
        with open("metrics/summarization/" + self.name +  "_failures.txt", "w") as f:
            for failure in self.failures:
                f.write(f"{failure}\n")

        with open("metrics/summarization/" + self.name + "_baseline_failures.txt", "w") as f:
            for failure in self.baseline_failures:
                f.write(f"{failure}\n")

        with open("metrics/summarization/" + self.name + "_metrics.txt", "w") as f:
            self.failure_mean = sum(self.failure_scores) / max(1, len(self.failure_scores))
            self.baseline_mean = sum(self.baseline_scores) / max(1, len(self.baseline_scores))
            self.emd = wasserstein_distance(np.array(self.failure_scores), np.array(self.baseline_scores))

            f.write(f"Failure Score Mean: {self.failure_mean}\n")
            f.write(f"Failure Score Std: {statistics.pstdev(self.failure_scores)}\n")
            f.write(f"Baseline Score Mean: {self.baseline_mean}\n")
            f.write(f"Baseline Score Std: {statistics.pstdev(self.baseline_scores)}\n")
            f.write(f"Earth Movers Distance: {self.emd}\n")

        plt.style.use('seaborn-deep')
        plt.hist(self.failure_scores, np.linspace(1.0, 10.0, 20), alpha = 0.5, label = 'transfer')
        plt.hist(self.baseline_scores, np.linspace(1.0, 10.0, 20), alpha = 0.5, label = 'baseline')
        plt.legend(loc = 'upper right')
        plt.title("Summarization Score Failure Transfer")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.savefig("metrics/" + self.name + "_histogram.png")
        plt.close()
        
        return (self.name, len(self.failures) / len(self.initial_domain), len(self.baseline_failures) / len(self.baseline), self.failure_mean, self.baseline_mean, self.emd)
