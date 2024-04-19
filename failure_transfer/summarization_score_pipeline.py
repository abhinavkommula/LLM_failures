import numpy as np
import random, math, statistics
import spacy
import re
import os
import nltk

from task import Task
from ast import literal_eval
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SummarizationScore(Task):
    def __init__(self, failure_mode, num_examples, interacter, logger, initial_domain = [], name = "score_task", domain = ("", None), read_file = None):
        super().__init__(name, failure_mode, num_examples, interacter, logger, initial_domain, read_file)
        
        self.domain = domain[0]
        self.scraper = domain[1]     

    def gen_data(self):
        if self.read_file != None:            
            self.initial_domain = []
            self.baseline = ["..."]
            
            with open(self.read_file, 'r') as f:
                for line in f:
                    cleaned_line = line.replace('\n', '')
                    
                    self.initial_domain.append(cleaned_line)
                                        
            return

        elif len(self.initial_domain) == 0:
            self.logger.print("Num examples: " + str(self.num_examples))
            self.initial_domain = self.scraper.get_examples(self.num_examples)
            self.logger.print(str(len(self.initial_domain)))
            self.baseline = ["..."]
             
        else:
            self.baseline = ["..."] 
        
    def score_summary(self, original, summary,  model = 'gpt-4-turbo'):
        start = f'You will be given a summary SUMMARY and original text ORIGINAL, and you will be asked to score the quality of the summary on a scale from 0 to 1. A strong summary should contain all the important points from the original text, and does not introduce new points that are not supported by the original text. '
        
        #start = f'You will be given a summary SUMMARY and original text ORIGINAL, and you will be asked to score the quality of the summary on a scale from 1 to 10. A strong summary should have grammatically correct sentences, should contain all the important points from the original text, and make logical sense. '
        end = f'Rate how strong SUMMARY is on a scale from 0 to 1 where 1 represents a strong summary while a 0 represents an incorrectly-written summary. Be a harsh critic when giving a score. Respond only with an integer from 0 to 1. '
        prompt = f'{start}\nORIGINAL:\n{original}\nSUMMARY:\n{summary}\n{end}'
    
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        scores = self.run_gpt(messages, model, max_tokens = 5, temperature = 0  )

        try:
            scores = literal_eval(scores)
        except:
            scores = 0

        return (scores)
                
    def pipeline(self):
        summarize_prefix = f"Summarize the following text without losing significant meaning. Text: [\n" 
        summarize_suffix = f"]\nSummarize all of the text provided into 2 sentences. Do not include any explanation." 
        
        def extract_answers(response):
            return ([response])

        input_domain = self.initial_domain
        all_scores = []
        all_failures = []

        for iterations in range(2):
            questions = []
            for i in range(len(input_domain)):
                questions.append(summarize_prefix + input_domain[i] + summarize_suffix)
            answers = self.interacter.answer_questions(questions, extract_answers)
                        
            assert(len(questions) == len(answers))

            failures = []
            scores = []

            for i in range(len(questions)):
                failures.append((input_domain[i], answers[i], self.score_summary(input_domain[i], answers[i])))

            input_domain = self.baseline
            all_scores.append(scores)

            all_failures.append(failures)

        self.failure_scores = all_scores[0]
        self.baseline_scores = all_scores[1]

        self.failures = all_failures[0]
        self.baseline_failures = all_failures[1]

    def extract_metrics(self):
        with open("metrics/summarization/data/" + self.name +  "_failures.txt", "w") as f:
            self.failures.sort(key = lambda t : len(t[0]))

            for failure in self.failures:
                f.write(f"{failure}\n")

        if len(self.failure_scores) == 0 or len(self.baseline_scores) == 0:
            self.failure_scores.append(0)
            self.baseline_scores.append(0)

        '''
        with open("metrics/summarization/data/" + self.name + "_metrics.txt", "w") as f:
            #self.failure_mean = sum(self.failure_scores) / max(1, len(self.failure_scores))
            #self.baseline_mean = sum(self.baseline_scores) / max(1, len(self.baseline_scores))
            #self.emd = wasserstein_distance(np.array(self.failure_scores), np.array(self.baseline_scores))

            #f.write(f"Failure Score Mean: {self.failure_mean}\n")
            #f.write(f"Failure Score Std: {statistics.pstdev(self.failure_scores)}\n")
            #f.write(f"Baseline Score Mean: {self.baseline_mean}\n")
            #f.write(f"Baseline Score Std: {statistics.pstdev(self.baseline_scores)}\n")
            #f.write(f"Earth Movers Distance: {self.emd}\n")

            f.write(f"Precision Mean: {statistics.mean(self.precision)}\n")
            f.write(f"Precision Std: {statistics.pstdev(self.precision)}\n")
            #f.write(f"Recall Mean: {statistics.mean(self.recall)}\n")
            #f.write(f"Recall Std: {statistics.pstdev(self.recall)}\n")
            #f.write(f"Rouge F1 Mean: {statistics.mean(self.rouge)}\n")
            #f.write(f"Rouge F1 Std: {statistics.pstdev(self.rouge)}\n")
            #f.write(f"Meteor Mean: {statistics.mean(self.meteor)}\n")
            #f.write(f"Meteor Std: {statistics.pstdev(self.meteor)}\n")
            
        plt.style.use('seaborn-deep')
        plt.hist(self.failure_scores, np.linspace(1.0, 10.0, 20), alpha = 0.5, label = 'transfer')
        plt.hist(self.baseline_scores, np.linspace(1.0, 10.0, 20), alpha = 0.5, label = 'baseline')
        plt.legend(loc = 'upper right')
        plt.title("Summarization Score Failure Transfer")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.savefig("metrics/summarization/data/" + self.name + "_histogram.png")
        plt.close()
        '''
        
        return (self.name, -1, -1)
        #return (self.name, len(self.failures) / len(self.initial_domain), len(self.baseline_failures) / len(self.baseline), self.failure_mean, self.baseline_mean, self.emd)
