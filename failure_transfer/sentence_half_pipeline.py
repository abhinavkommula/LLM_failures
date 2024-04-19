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

class SentenceHalf(Task):
    def __init__(self, failure_mode, num_examples, interacter, logger, initial_domain = [], name = "score_task", read_file = None):
        super().__init__(name, failure_mode, num_examples, interacter, logger, initial_domain, read_file)

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
        half_prefix = "I will provide you with two lists, each containing 5 half-sentences. For each half-sentence in LIST1, combine it with exactly one half-sentence in LIST2 that is the most likely to complete the sentence. You may not match the same half-sentence in LIST2 with multiple different half-sentences in LIST1."
        half_suffix = ". Two combined half-sentences are called a full-sentence. Format your output as: '1. Full-Sentence, 2. Full-Sentence, 3. Full-Sentence, 4. Full-Sentence, 5. Full-Sentence', and do not include any explanation or additional dialogue. "
        
        def extract_answers(response):
            response = re.sub(r'\(.*?\)', '', response)
            result = [re.split(r'(?<=\d\.)\s*', response)[1:]]

            for i in range(len(result[0])):
                if type(result[0][i]) == str:
                    result[0][i] = re.sub(r'\b\d+\.\b', '', result[0][i])
                
            return (result)

        def derive_half_lists(paragraph):
            sentences = re.split(r'(?<=[.!?])+', paragraph)
            sentences.sort(key = lambda s : len(s))
            selected_sentences = sentences[-5:]

            first_halves = []
            second_halves = []

            for sentence in selected_sentences:
                words = sentence.split()
                middle_index = len(words) // 2

                if len(sentence) % 2 != 0 and len(words[middle_index]) > 1:
                    middle_index += 1

                first_half = ' '.join(words[:middle_index])
                second_half = ' '.join(words[middle_index:])

                first_halves.append(first_half)
                second_halves.append(second_half)
            
            first_halves_shuffled = first_halves.copy()
            random.shuffle(first_halves_shuffled)
            
            second_halves_shuffled = second_halves.copy()
            random.shuffle(second_halves_shuffled)  
            
            list1_str = ", ".join([f'"{half}"' for half in first_halves_shuffled])
            list2_str = ", ".join([f'"{half}"' for half in second_halves_shuffled])
            return f"LIST1: [{list1_str}]\n, LIST2: [{list2_str}]", first_halves, second_halves
            
        def half_similarity(original, full_sentences):
            sentences_orig = re.split(r'(?<=[.!?])+', original)

            vectorizer = CountVectorizer()
            found = 0
            
            for i, full_sentence in enumerate(full_sentences):
                all_sentences = [full_sentence] + sentences_orig
                bow_matrix = vectorizer.fit_transform(all_sentences)
                similarities = cosine_similarity(bow_matrix[0:1], bow_matrix[1:])
                
                if np.any(similarities >= 0.7):
                    found += 1
            
            print("Response:", full_sentences)
            print("Original:", original)
            print("Similarity:", found/len(sentences_orig))
            
            return (found / len(sentences_orig))
                
        input_domain = self.initial_domain
        all_scores = []
        all_failures = []

        for iterations in range(1):
            questions = []
            half_lists = []
            for i in range(len(input_domain)):
                result_str, list_1, list_2 = derive_half_lists(input_domain[i])
                half_lists.append((list_1, list_2))
                questions.append(half_prefix + result_str + half_suffix)
                                
            answers = self.interacter.answer_questions(questions, extract_answers)
                        
            assert(len(questions) == len(answers))

            failures = []
            scores = []
                        
            for i in range(len(answers)):
                failures.append((input_domain[i], answers[i], half_lists[i][0], half_lists[i][1], half_similarity(input_domain[i], answers[i])))

            input_domain = self.baseline
            all_scores.append(scores)

            all_failures.append(failures)

        self.failure_scores = all_scores[0]
        #self.baseline_scores = all_scores[1]

        self.failures = all_failures[0]
        #self.baseline_failures = all_failures[1]

    def extract_metrics(self):
        with open("metrics/sentence_half/data/" + self.name +  "_failures_similarity.txt", "w") as f:
            self.failures.sort(key = lambda f : f[4]) 
            self.similarity = [f[4] for f in self.failures]

            for failure in self.failures:
                f.write(f"{failure}\n")
                                
        with open("metrics/sentence_half/data/" + self.name + "_metrics.txt", "w") as f:
            f.write(f"Similarity Mean: {statistics.mean(self.similarity)}\n")
            f.write(f"Similarity Std: {statistics.pstdev(self.similarity)}\n") 
                       
        return (self.name, -1, -1)
