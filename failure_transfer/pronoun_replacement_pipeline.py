import numpy as np
import random, math, statistics
import spacy, torch
import re
import os

from task import Task
from ast import literal_eval

import nltk
import re
import spacy

nlp = spacy.load('en_core_web_sm')

class PronounReplacement(Task):
    def __init__(self, failure_mode, num_examples, interacter, logger, initial_domain = [], name = "pronoun_replacement_task", read_file = None):
        super().__init__(name, failure_mode, num_examples, interacter, logger, initial_domain, read_file)

    def gen_data(self):
        self.baseline = [] 

    def pipeline(self):
        pronoun_replacement_prefix = "Given the following text, replace all pronouns with their correct antecedents. Text: [\n"
        pronoun_replacement_suffix = "]\nReplace all pronouns in the text with the corresponding noun or noun substitute (antecedent) that the pronoun refers to. Only replace the pronouns, do not modify the text in any other manner. Your output should contain no pronouns. Format your text as 'Text: ...', and do not include any explanation. "
        
        def extract_answers(response):
            if "Text:" in response:
                return [response.split("Text:")[1]]
            
            return [response]
        
        def get_antecedent(doc, pronoun):
            for token in doc:
                if token.text.lower() == pronoun.lower():
                    # Case 1: Pronoun is a subject
                    if token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass':
                        # Check for auxiliary verbs
                        if token.head.pos_ == 'AUX' and token.head.head.pos_ == 'VERB':
                            verb = token.head.head
                        else:
                            verb = token.head
                        for child in verb.children:
                            if child.dep_ == 'dobj' or child.dep_ == 'attr' or child.dep_ == 'pobj':
                                return child.text
                            if child.dep_ == 'prep' and child.text.lower() == 'to':
                                for grandchild in child.children:
                                    if grandchild.dep_ == 'pobj':
                                        return grandchild.text

                    # Case 2: Pronoun is an object
                    elif token.dep_ == 'dobj' or token.dep_ == 'pobj':
                        subject = None
                        for child in token.head.children:
                            if child.dep_ == 'nsubj' or child.dep_ == 'nsubjpass':
                                subject = child
                                break
                        if subject:
                            return subject.text

                    # Case 3: Pronoun is a possessive modifier
                    elif token.dep_ == 'poss':
                        return token.head.text

                    # Case 4: Pronoun follows a linking verb (e.g., is, are, was, were)
                    elif token.dep_ == 'attr' and token.head.pos_ == 'VERB':
                        for child in token.head.children:
                            if child.dep_ == 'nsubj' or child.dep_ == 'nsubjpass':
                                return child.text

            return None
        
        def evaluate_pronoun(original_paragraph, generated_paragraph):
            nlp = spacy.load('en_core_web_sm')
            original_doc = nlp(original_paragraph)
            generated_doc = nlp(generated_paragraph)
            
            original_pronouns = [token.text.lower() for token in original_doc if token.pos_ == 'PRON']
            generated_pronouns = [token.text.lower() for token in generated_doc if token.pos_ == 'PRON']
            
            penalty = 0
            total = 0
            
            for pronoun in original_pronouns:
                original_antecedent = get_antecedent(original_doc, pronoun)
                generated_antecedent = get_antecedent(generated_doc, pronoun)
                
                if original_antecedent != generated_antecedent:
                    penalty += 1
                
                total += 1
            
            return (penalty / max(1, total))
        

        input_domain = self.initial_domain
        all_failures = []
        failures = []

        for iteration in range(1):
            questions = []
            for i in range(len(input_domain)):
                questions.append(pronoun_replacement_prefix + input_domain[i] + pronoun_replacement_suffix)
                
            answers = self.interacter.answer_questions(questions, extract_answers)
                    
            assert(len(questions) == len(answers))
                    
            for i in range(len(input_domain)):         
                failures.append((input_domain[i], answers[i], evaluate_pronoun(input_domain[i], answers[i])))

            input_domain = self.baseline
            all_failures.append(failures)

        self.failures = all_failures[0]
        #self.baseline_failures = all_failures[1]

    def extract_metrics(self):
        with open("metrics/pronoun_replacement/data/" + self.name +  "_failures.txt", "w") as f:
            for failure in self.failures:
                f.write(f"{failure}\n")
                       
        return (self.name, -1, -1)
