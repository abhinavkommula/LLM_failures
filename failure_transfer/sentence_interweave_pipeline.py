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
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import sent_tokenize
from rouge import Rouge
from collections import Counter

nlp = spacy.load("en_core_web_md")
nltk.download("wordnet")
nltk.download("punkt")
rouge_obj = Rouge()

class SentenceInterweave(Task):
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
        interleave_prefix = "I have 2 unrelated paragraphs, and I have interwoven their sentences together. Do your best to recover original two paragraphs, Paragraph1 and Paragraph2. I have provided the first few sentences from Paragraph1 and Paragraph2.\n"
        interleave_middle = "\nInterwoven Text: [\n"
        interleave_suffix = "]\nComplete Paragraph1 and Paragraph2 using sentences from Interwoven Text. Format your output as: 'Paragraph1: ..., Paragraph2: ...', and do not include any explanation or additional    ."

        def extract_answers(response):
            paragraphs = re.split(r'(Paragraph\s*[A-Z0-9]+)', response)            
            return [paragraphs[1:]]
        
        def separate_sentences(text):
            sentences = []
            current_sentence = ""
            in_quotes = False
            
            for char in text:
                if char in ['"', "'"]:
                    in_quotes = not in_quotes
                    
                current_sentence += char
                
                if char in ['.', '?', '!', '\n'] and not in_quotes:
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
                    
            if current_sentence:
                sentences.append(current_sentence.strip())
                
            return sentences

        def gen_middle(paragraph1, paragraph2):
            sentences1 = separate_sentences(paragraph1)
            sentences2 = separate_sentences(paragraph2)
            
            str = "Paragraph1: [\n"

            for i in range(min(3, len(sentences1) - 1)):
                str += sentences1[i]
    
            str += "\n]"
            str += "Paragraph2: [\n"
            
            for i in range(min(3, len(sentences2) - 1)):
                str += sentences2[i]
             
            str += "\n]"
            return (str)
            
        def interleave_sentences(paragraph1, paragraph2):
            sentences1 = separate_sentences(paragraph1)
            sentences2 = separate_sentences(paragraph2)

            interleaved_sentences = []
            while sentences1 and sentences2:
                if random.choice([True, False]):  
                    interleaved_sentences.append(sentences1.pop(0))
                else: 
                    interleaved_sentences.append(sentences2.pop(0))

            interleaved_sentences.extend(sentences1 or sentences2)

            interleaved_paragraph = ' '.join(interleaved_sentences)
            return interleaved_paragraph

        def interleave_similarity(original, paragraph):
            sentences1 = separate_sentences(original)
            sentences2 = separate_sentences(paragraph)

            vectorizer = CountVectorizer()
            found = 0
            
            for i, sentence1 in enumerate(sentences1):
                all_sentences = [sentence1] + sentences2
                
                try:
                    bow_matrix = vectorizer.fit_transform(all_sentences)
                    similarities = cosine_similarity(bow_matrix[0:1], bow_matrix[1:])
                                
                    if np.any(similarities >= 0.7):
                        found += 1
                except:
                    continue
            
            return (found / len(sentences1))
                
        input_domain = self.initial_domain
        all_scores = []
        all_failures = []

        for iterations in range(1):
            questions = []
            randomized = []
            interleaved = []

            for i in range(len(input_domain)):
                rand_choice = random.choice(input_domain)
                
                while (rand_choice == input_domain[i]):
                    rand_choice = random.choice(input_domain)
                
                randomized.append(rand_choice)
                interleaved.append(interleave_sentences(input_domain[i], randomized[i]))
                questions.append(interleave_prefix + gen_middle(input_domain[i], randomized[i]) + interleave_middle + interleaved[-1] + interleave_suffix)
                
            answers = self.interacter.answer_questions(questions, extract_answers)
                    
            assert(len(questions) == len(answers))

            failures = []
            scores = []
                        
            for i in range(len(answers)):
                extract = ["", ""]
                text_1 = False
                
                for cur_str in answers[i]:
                    if "Paragraph1" in cur_str:
                        text_1 = True
                    elif "Paragraph2" in cur_str:
                        text_1 = False
                    else:
                        if text_1:
                            extract[0] += cur_str
                        else:
                            extract[1]  += cur_str
                                
                if len(extract[0]) == 0:
                    extract[0] = "Emtpy"
                if len(extract[1]) == 0:
                    extract[1] = "Empty"
                                
                failures.append((input_domain[i], randomized[i], interleaved[i], extract, interleave_similarity(input_domain[i], extract[0]), interleave_similarity(randomized[i], extract[1])))

            input_domain = self.baseline
            all_scores.append(scores)

            all_failures.append(failures)

        self.failure_scores = all_scores[0]
        #self.baseline_scores = all_scores[1]

        self.failures = all_failures[0]
        #self.baseline_failures = all_failures[1]

    def extract_metrics(self):
        with open("metrics/sentence_interweave/data/" + self.name +  "_failures_similarity.txt", "w") as f:
            self.failures.sort(key = lambda f : (f[4] + f[5]) / 2) 
            self.similarity = [(f[4] + f[5]) / 2 for f in self.failures]

            for failure in self.failures:
                f.write(f"{failure}\n")
                
        print ("Similarity:", self.similarity)
                                
        with open("metrics/sentence_interweave/data/" + self.name + "_metrics.txt", "w") as f:
            f.write(f"Similarity Mean: {statistics.mean(self.similarity)}\n")
            f.write(f"Similarity Std: {statistics.pstdev(self.similarity)}\n") 
                       
        return (self.name, -1, -1)
