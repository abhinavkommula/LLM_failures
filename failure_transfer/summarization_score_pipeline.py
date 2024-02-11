from task import Task
from ast import literal_eval
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance

import numpy as np
import random, math, statistics
import re
import os

class SummarizationScore(Task):
    def __init__(self, failure_mode, num_examples, interacter, name = "score_task", domain = ("", None), read_file = None):
        super().__init__(name, failure_mode, num_examples, interacter, read_file)
        
        self.domain = domain[0]
        self.scraper = domain[1]
   
    def domain_transfer(self, paragraph, model = 'gpt-3.5-turbo'):
        start = f'Given a paragraph PARAGRAPH, please rewrite it in the style of a {self.domain}. You will be evaluated on how well you perform. Here are examples from this domain as a reference.'
        end = f'Ensure that all important points in the paragraph are contained your response. Your sentence structure and length can be creative. Be creative and cautious. Limit the length of your response to 5 sentences.' 
        prompt = f'{start}\nEXAMPLE #1:{self.scraper.get_random()}\nEXAMPLE #2:{self.scraper.get_random()}\nPARAGRAPH\n{paragraph}\n{end}'

        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        output_domain = self.run_gpt(messages, model, max_tokens = 1000, temperature = 0)
        output_domain = output_domain.replace('\n', '')
       
        return (output_domain)

    def gen_data(self):
        if self.read_file != None:
            paragraphs_list = []

            with open(self.read_file, 'r') as f:
                for line in f:
                    tup = literal_eval(line)
                    paragraphs_list.append((tup[0], tup[2], tup[1]))

            idx = 5
            while paragraphs_list[idx][1] <= 6:
                idx += 1

            print("Mean failure transfer:", statistics.mean([el[1] for el in paragraphs_list[:idx]]))
            
            random.shuffle(paragraphs_list)
            print("Mean baseline:", statistics.mean([el[1] for el in paragraphs_list[:30]]))

            self.initial_domain = [self.domain_transfer(el[0]) for el in paragraphs_list[:idx]]
            self.baseline = [self.domain_transfer(el[0]) for el in paragraphs_list[:30]]

            return

        #prompt = "Write down 3 separate and unrelated paragraphs in the style of {self.domain}. You will be evaluated on how well you perform. Your sentence structure and length can be creative. Please format your numbered output as: Paragraph 1:, Paragraph 2:, ..."

        self.initial_domain = self.scraper.get_examples(self.num_examples)
        #self.initial_domain = self.gen_failures(context = prompt, num_paragraphs = 3)

        self.baseline = ["..."] 
        #self.baseline = self.gen_failures(context = baseline_prompt, num_paragraphs = 3, is_baseline = True)

    def score_summary(self, summary, document, model = 'gpt-3.5-turbo'):
        start = f'You will be given a summary SUMMARY for a document DOCUMENT, and you will be asked to score the quality of the summary on a scale from 1 to 10. A strong summary should be as succinct as possible, contain only details necessary to understanding DOCUMENT, and have logical transitions between ideas. '
        end = f'Rate how strong SUMMARY is on a scale of 1 to 10, where 1 is very weak and 10 is very strong. Respond only with a number from 1 to 10. '
        prompt = f'{start}\nSUMMARY:\n{summary}\nDOCUMENT:\n{document}\n{end}'
    
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        scores = self.run_gpt(messages, model, max_tokens = 1000, temperature = 0)

        try:
            scores = literal_eval(scores)
        except:
            scores = 0

        return (scores)

    def pipeline(self):
        summarize_prefix = f"Summarize the following text as succinctly as possible. Capture only the points necessary to understanding the text. Only write 1 sentence. "  
        
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
                questions.append(summarize_prefix + input_domain[i])
            answers = self.interacter.answer_questions(questions, extract_answers)
            
            assert(len(questions) == len(answers))

            failures = []
            scores = []

            for i in range(len(questions)):
                scores.append(self.score_summary(answers[i], input_domain[i]))
   
                # Modify if we decide to apply thresholding
                if scores[-1] <= 10:
                    failures.append((input_domain[i], answers[i], scores[-1]))

            input_domain = self.baseline
            all_scores.append(scores)

            failures.sort(key = lambda f : f[2])
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
        plt.savefig("metrics/summarization/" + self.name + "_histogram.png")
        plt.close()
        
        return (self.name, len(self.failures) / len(self.initial_domain), len(self.baseline_failures) / len(self.baseline), self.failure_mean, self.baseline_mean, self.emd)
