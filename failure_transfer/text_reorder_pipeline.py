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

class TextReorder(Task):
    def __init__(self, failure_mode, num_examples, interacter, logger, initial_domain = [], name = "score_task", read_file = None):
        super().__init__(name, failure_mode, num_examples, interacter, logger, initial_domain, read_file)
            
    def gen_reorder(self, prompt, model = 'gpt-4-turbo'):
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        output_reorder = self.run_gpt(messages, model, max_tokens = 1000, temperature = 0)
        output_reorder = output_reorder.replace('\n', '')
       
        return (output_reorder)        

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
        
    def find_entities(self, text):
        doc = nlp(text)
        entities = [ent.text.lower() for ent in doc.ents]
        
        for chunk in doc.noun_chunks:
            head_text = chunk.root.head.text
            if chunk.root.head.pos_ == 'ADJ' and head_text not in chunk.text:
                entities.append(f"{head_text.lower()} {chunk.text.lower()}")
            else:
                entities.append(chunk.text.lower())

        return list(set(entities))

    def find_similar(self, entities1, entities2):
        if len(entities2) > len(entities1):
            return (-1.0, -1.0)

        found_orig = {}
        found_modified = {}
        modified_map = {}

        for s in entities2:
            modified_map[s] = True

        for o in entities1:
            if o in modified_map:
                found_orig[o] = True
                found_modified[o] = True
        
        true_positives = 0
        false_negatives = 0
        false_positives = 0

        for s in entities2:
            if s in found_modified:
                true_positives += 1
            else:
                false_positives += 1

        for o in entities1:
            if o not in found_orig:
                false_negatives += 1

        return (true_positives / max(1, (true_positives + false_positives)), true_positives / max(1, (true_positives + false_negatives))) 
                
    def pipeline(self):
        reorder_prefix = f"Given a list of unordered sentences, please reorder them into a paragraph with logical transitions. Remember to include all of the sentences in your output, and do not combine sentences together. The first three sentences are in the correct order. List of sentences: "  
        reorder_suffix = f"\nThe first 3 sentences are in correct order. Format your output as a coherent paragraph with the same number of sentences as the original list of sentences. Your output should only contain an ordering of the original sentences, do not modify them in any way. Format your output as 'Reorder: ...', and do not include any explanation." 
        
        def extract_answers(response):
            if 'Reorder:' not in response:
                return ([response])
            
            return ([':'.join(response.split("Reorder:")[1:])])

        def separate_sentences(text):
            sentences = []
            current_sentence = ""
            in_quotes = False
            
            for char in text:
                if char in ['"', "'"]:
                    in_quotes = not in_quotes
                    
                current_sentence += char
                
                if char in ['.', '?', '!'] and not in_quotes:
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
                    
            if current_sentence:
                sentences.append(current_sentence.strip())
                
            return sentences

        def change_to_reorder(paragraph):
            result = "[\n"
            
            sentences = separate_sentences(paragraph)
            rest_sentences = sentences[3:]
            random.shuffle(rest_sentences)
            sentences = sentences[:3] + rest_sentences
            
            result += "\n".join(f"{i + 1}. {sentence}" for i, sentence in enumerate(sentences))
            result += "\n]"
            return (result)
        
        def reorder_sim(paragraph1, paragraph2):
            sentences1 = nltk.sent_tokenize(paragraph1)
            sentences2 = nltk.sent_tokenize(paragraph2)

            vectorizer = CountVectorizer()

            all_sentences = sentences1 + sentences2
            vectorizer.fit(all_sentences)

            vectors1 = vectorizer.transform(sentences1).toarray()
            vectors2 = vectorizer.transform(sentences2).toarray()

            inversions = 0
            closest_indices = []
            for vec2 in vectors2:
                try:
                    similarities = cosine_similarity([vec2], vectors1)
                    closest_index = np.argmax(similarities)

                    if len(closest_indices) > 0 and closest_index < closest_indices[-1]:    
                        inversions += 1
                
                    closest_indices.append(closest_index)
                except:
                    continue

            return (inversions / max(1, (len(closest_indices))))

        input_domain = self.initial_domain
        all_scores = []
        all_failures = []

        for iterations in range(1):
            questions = []
            for i in range(len(input_domain)):
                questions.append(reorder_prefix + change_to_reorder(input_domain[i]) + reorder_suffix)
            answers = self.interacter.answer_questions(questions, extract_answers)
            
            assert(len(questions) == len(answers))

            failures = []
            scores = []
                        
            for i in range(len(answers)):
                failures.append((input_domain[i], answers[i], reorder_sim(input_domain[i], answers[i])))

            input_domain = self.baseline
            all_scores.append(scores)

            all_failures.append(failures)

        self.failure_scores = all_scores[0]
        #self.baseline_scores = all_scores[1]

        self.failures = all_failures[0]
        #self.baseline_failures = all_failures[1]

    def extract_metrics(self):
        with open("metrics/text_reorder/data/" + self.name +  "_failures_similarity.txt", "w") as f:
            self.failures.sort(key = lambda f : -f[2]) 
            self.similarity = [t[2] for t in self.failures]

            for failure in self.failures:
                f.write(f"{failure}\n")
                                
        with open("metrics/text_reorder/data/" + self.name + "_metrics.txt", "w") as f:
            f.write(f"Similarity Mean: {statistics.mean(self.similarity)}\n")
            f.write(f"Similarity Std: {statistics.pstdev(self.similarity)}\n") 
                       
        return (self.name, -1, -1)
