import numpy as np
import random, math, statistics
import spacy, torch
import re
import os
import nltk

from task import Task
from ast import literal_eval
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from collections import Counter

class PartOfSpeech(Task):
    def __init__(self, failure_mode, num_examples, interacter, logger, initial_domain = [], name = "score_task", read_file = None, pos_type = "noun"):
        super().__init__(name, failure_mode, num_examples, interacter, logger, initial_domain, read_file)
        
        self.pos_type = pos_type

    def gen_data(self):
        if self.read_file != None:            
            self.initial_domain = []
            self.failure_mode_idx = []
            self.baseline = ["..."]
            
            cur_failure_mode = -1            
            with open(self.read_file, 'r') as f:
                for line in f:
                    cleaned_line = line.replace('\n', '')
                        
                    if "FAILURE MODE ->" in cleaned_line:
                        cur_failure_mode += 1
                        continue

                    self.initial_domain.append(cleaned_line)
                    self.failure_mode_idx.append(cur_failure_mode)
                    
            return

        elif len(self.initial_domain) == 0:
            self.logger.print("Num examples: " + str(self.num_examples))
            self.initial_domain = self.scraper.get_examples(self.num_examples)
            self.logger.print(str(len(self.initial_domain)))
            self.baseline = ["..."]
             
        else:
            self.baseline = [] 
    
    def pipeline(self):
        pos_prefix = "Please fill in the blanks indicated by the string '___' to the best of your ability. Respond with the entire completed paragraph, and nothing more. Keep the structure of the sentences intact in your output. Text: "
        
        def extract_answers(response):
            if ':' not in response:
                return ([response])
            
            return ([':'.join(response.split(":")[1:])])

        def remove_pos(paragraph, word_type):
            sentences = re.split(r'(?<=[.!?]) +', paragraph)
            modified_sentences = []

            for sentence in sentences:
                doc = nlp(sentence)
                words = [token.text for token in doc]
                indices_of_type = [i for i, token in enumerate(doc) if token.pos_ == word_type.upper()]

                if indices_of_type:
                    random.shuffle(indices_of_type)
                    words[indices_of_type[0]] = "___"
                
                modified_sentences.append(' '.join(words))

            modified_paragraph = ' '.join(modified_sentences)
            return modified_paragraph

        def sentence_similarity(paragraph1, paragraph2):
            sentences1 = re.split(r'(?<=[.!?]) +', paragraph1)
            sentences2 = re.split(r'(?<=[.!?]) +', paragraph2)

            vectorizer = CountVectorizer()
            total_similarity = 0

            for sentence1 in sentences1:
                all_sentences = [sentence1] + sentences2
                bow_matrix = vectorizer.fit_transform(all_sentences)
                similarities = cosine_similarity(bow_matrix[0:1], bow_matrix[1:])

                if similarities.size > 0:
                    max_similarity = np.max(similarities)
                    total_similarity += max_similarity

            average_similarity = total_similarity / len(sentences1) if sentences1 else 0
            return average_similarity

        input_domain = self.initial_domain
        all_scores = []
        all_failures = []

        for iterations in range(1):
            questions = []
            removed = []
            
            for i in range(len(input_domain)):
                rem = remove_pos(input_domain[i], self.pos_type)    
                removed.append(rem)
                questions.append(pos_prefix + rem)
                
            answers = self.interacter.answer_questions(questions, extract_answers)
            
            assert(len(questions) == len(answers))

            failures = []
            scores = []
                        
            for i in range(len(answers)):
                failures.append((input_domain[i], removed[i], answers[i], sentence_similarity(input_domain[i], answers[i])))

            input_domain = self.baseline
            all_scores.append(scores)

            all_failures.append(failures)

        self.failure_scores = all_scores[0]
        #self.baseline_scores = all_scores[1]

        self.failures = all_failures[0]
        #self.baseline_failures = all_failures[1]

    def extract_metrics(self):
        with open("metrics/pos/data/" + self.name +  "_failures_similarity.txt", "w") as f:
            self.failures.sort(key = lambda f : f[3]) 
            self.similarity = [t[3] for t in self.failures]

            for failure in self.failures:
                f.write(f"{failure}\n")
                                
        with open("metrics/pos/data/" + self.name + "_metrics.txt", "w") as f:
            f.write(f"Similarity Mean: {statistics.mean(self.similarity)}\n")
            f.write(f"Similarity Std: {statistics.pstdev(self.similarity)}\n") 
                       
        return (self.name, -1, -1)
