from task import Task
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

import random, math
import re
import os

class Reasoning(Task):
    def __init__(self, failure_mode, num_examples, interacter):
        super().__init__("reasoning_task", failure_mode, num_examples, interacter)
        self.scrape_question_answers()
    
    def scrape_question_answers(self):
         dataset = load_dataset("tau/commonsense_qa")

         self.m_questions = []
         self.m_answers = []
         self.answer_choices_str = []

         for card in dataset['train']:
            d_choices = card['choices']
            answer_str = ""    
        
            for i in range(len(d_choices['label'])):
                answer_str += (d_choices['label'][i] + ". " + d_choices['text'][i] + ", ")

            self.m_questions.append(card['question'])
            self.m_answers.append(card['answerKey'])
            self.answer_choices_str.append(answer_str[:-2])
            
            
    def gen_data(self):
        prompt = "Imagine you are a language model with the following failure mode that is tasked with answering reasoning questions. Please rewrite the following question so that it would likely elicit failures in your ability to answer reasoning questions. You will be evaluated on how well you perform. Extrapolate based on the failure mode, but do not change the fundamental meaning of the original question. Question: "

        baseline_prompt = "Imagine you are a language model that is tasked with answering reasoning questions. Please rewrite the following question so that it would likely elicit failures in your ability to answer reasoning questions. You will be evaluated on how well you perform. Extrapolate as you see fit, but do not change the fundamental meaning of the original question. Question: "

        self.initial_domain = self.gen_failures(context = prompt, num_paragraphs = 1)
        self.baseline = self.gen_failures(context = baseline_prompt, num_paragraphs = 1, is_baseline = True)

    def gen_failures(self, context, num_paragraphs = 1, is_baseline = False, model = 'gpt-3.5-turbo'):
        iteration_number = self.num_examples

        if is_baseline:
            iteration_number = (iteration_number // 2)

        failures = []
        
        for i in range(int(iteration_number)):
            if is_baseline:
                query = context + self.m_questions[i] + ". Only write the question"
            else:
                query = context + self.m_questions[i] + "\n" + "Failure Mode: [" + self.failure_mode + "]. Only write the question"
            
            messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': query}]
            llm_output = self.run_gpt(messages, model, max_tokens = 1000, temperature = 0.3)
            failures.append(llm_output)
            
            print("Failures Generated:", num_paragraphs * (i + 1))

        return (failures)

    def pipeline(self):
        reasoning_suffix = " . Respond to the best of your ability. Only respond with the letter of your answer."

        def extract_answers(response):
            return ([response])

        input_domain = self.initial_domain
        all_failures = []

        for iterations in range(2):
            questions = []
            
            for i in range(len(input_domain)):
                questions.append(input_domain[i] + " " + self.answer_choices_str[i] + reasoning_suffix) 
                print(questions[-1])

            answers = self.interacter.answer_questions(questions, extract_answers)

            failures = []
            for i in range(len(answers)):
                if answers[i][0] != self.m_answers[i]:
                    failures.append((self.m_questions[i], questions[i], answers[i][0], self.m_answers[i]))
            
            input_domain = self.baseline
            all_failures.append(failures)

        self.failures = all_failures[0]
        self.baseline_failures = all_failures[1]
