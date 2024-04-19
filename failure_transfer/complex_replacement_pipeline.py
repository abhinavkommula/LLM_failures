import numpy as np
import random, math, statistics
import spacy, torch
import re
import os

from task import Task
from ast import literal_eval

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.metrics import edit_distance

nltk.download('stopwords')

class ComplexReplacement(Task):
    def __init__(self, failure_mode, num_examples, interacter, logger, initial_domain = [], name = "complex_replacement_task", read_file = None):
        super().__init__(name, failure_mode, num_examples, interacter, logger, initial_domain, read_file)

    def gen_data(self):
        self.baseline = [] 

    def pipeline(self):
        complex_replacement_prefix = "Given the following text, please replace all instances of [BLANK] with a word that makes the most sense in context. Text: [\n"
        complex_replacement_suffix = "]\nReplace all instances of [BLANK] with the best word given the context of the paragraph. Your output should contain no [BLANK] instances. You should only replace instances of [BLANK] with a word, do not make any other modifications. Format your text as 'Text: ...', and do not include any explanation. "
        
        def extract_answers(response):
            if "Text:" in response:
                return [response.split("Text:")[1]]
            
            return [response]
        
        def is_complex_word(word, complexity_threshold=3):
            synsets = wordnet.synsets(word)
            if synsets:
                return max(len(synset.lemmas()) for synset in synsets) >= complexity_threshold
            return False
        
        def process_question(paragraph, num_blanks):
            words = paragraph.split()
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
            complex_word_indices = [i for i, word in enumerate(words) if is_complex_word(lemmatizer.lemmatize(word.lower())) and word.lower() not in stop_words]

            if len(complex_word_indices) >= num_blanks:
                blank_indices = []
                while len(blank_indices) < num_blanks:
                    index = random.choice(complex_word_indices)
                    if not blank_indices or ((index + 1) not in blank_indices and (index - 1) not in blank_indices):
                        blank_indices.append(index)
                        complex_word_indices.remove(index)
                
                for index in blank_indices:
                    words[index] = '[BLANK]'
            else:
                print(f"Not enough complex words found. Replacing {len(complex_word_indices)} words with blanks.")
                for index in complex_word_indices:
                    words[index] = '[BLANK]'
            
            return ' '.join(words)

        def find_closest_synonym(word, word_list):
            closest_word = None
            max_similarity = 0

            for candidate in word_list:
                similarity = 1 - (edit_distance(word, candidate) / max(len(word), len(candidate)))
                if similarity > max_similarity:
                    max_similarity = similarity
                    closest_word = candidate

            if closest_word:
                synsets_word = wordnet.synsets(word)
                synsets_closest = wordnet.synsets(closest_word)
                
                for synset_word in synsets_word:
                    for synset_closest in synsets_closest:
                        if synset_word == synset_closest:
                            return closest_word, True

            return closest_word, False
                
        def evaluate_filled(original_paragraph, modified_paragraph, filled_paragraph):
            original_words = original_paragraph.split()
            modified_words = modified_paragraph.split()
            filled_words = filled_paragraph.split()
            
            penalty = 0
            total = 0
            
            orig = []
            filled = []
            
            for i in range(len(original_words)):
                modified_words_sub = modified_words[max(0, i - 5) : min(len(modified_words), i + 5)]
                
                if original_words[i] not in modified_words_sub:
                    orig.append(original_words[i])
            
            for i in range(len(filled_words)):
                modified_words_sub = modified_words[max(0, i - 5) : min(len(modified_words), i + 5)]

                if filled_words[i] not in modified_words_sub:
                    filled.append(filled_words[i])
            
            for word in orig:
                ans = find_closest_synonym(word, filled)
                                
                if not ans[1]:
                    penalty += 1
                
                total += 1
                    
            return (penalty / max(1, total))

        input_domain = self.initial_domain
        all_failures = []
        failures = []

        for iteration in range(1):
            questions = []
            new_input = []
            
            for i in range(len(input_domain)):
                processed = process_question(input_domain[i], 10)
                new_input.append(processed)
                questions.append(complex_replacement_prefix + processed + complex_replacement_suffix)
                
            answers = self.interacter.answer_questions(questions, extract_answers)
                    
            assert(len(questions) == len(answers))
                    
            for i in range(len(input_domain)):         
                failures.append((input_domain[i], new_input[i], answers[i], evaluate_filled(input_domain[i], new_input[i], answers[i])))

            input_domain = self.baseline
            all_failures.append(failures)

        self.failures = all_failures[0]
        #self.baseline_failures = all_failures[1]

    def extract_metrics(self):
        with open("metrics/complex_replacement/data/" + self.name +  "_failures.txt", "w") as f:
            for failure in self.failures:
                f.write(f"{failure}\n")
                       
        return (self.name, -1, -1)
