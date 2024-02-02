from task import Task
from ast import literal_eval
from matplotlib import pyplot as plt

import numpy as np
import random, math
import re
import os

class SummarizationScore(Task):
    def __init__(self, failure_mode, num_examples, interacter):
        super().__init__("summarization_score_task", failure_mode, num_examples, interacter)

    def gen_data(self):
        prompt = "Imagine you are a language model with the following failure mode that is tasked with summarizing text. Write down 5 additional paragraphs that would likely elicit failures in your ability to summarize text. You will be evaluated on how well you perform. Your sentence structure and length can be creative; extrapolate based on the failure mode. Be both creative and cautious. "
        
        baseline_prompt = "Imagine you are a language model that is tasked with summarizing text. Write down 5 additional paragraphs that would likely elicit failures in your ability to summarize text. You will be evaluated on how well you perform. Your sentence structure and length can be creative; be both creative and cautious. "

        self.initial_domain = self.gen_failures(context = prompt, num_paragraphs = 5)
        self.baseline = self.gen_failures(context = baseline_prompt, num_paragraphs = 5, is_baseline = True)

    def score_summary(self, summary, model = 'gpt-3.5-turbo'):
        start = f'You will be given a summary SUMMARY, and your task will be to assess how strong of a summary it is on scale from 1 to 10. Here is an example of a summary that would recieve a score of 2, because a high proportion of it contains details that are irrelevant: [a $4 trillion package with revenues, a $1 trillion-$2 trillion spending cuts-only deal, or a deal that grants the President authority to raise the debt ceiling without GOP agreement. Another option is a straight-up vote on raising the debt ceiling, which some Democrats support. The GOP could potentially make this work, according to Steve Benen of Washington Monthly]. Here is an example of a summary that would receive a score of 7 because it has a smaller portion of irrelevant information and makes more logical sense: [Richard "The Old Man" Harrison, a star of the popular reality TV show "Pawn Stars," has passed away at the age of 77 due to complications from Parkinsons disease. He will be greatly missed by his family, the team at Gold & Silver Pawn, and his many fans around the world. Harrison was a Navy veteran who opened the Gold & Silver Pawn Shop in Las Vegas in 1988, which became a multimillion-dollar business after the show premiered in 2009. He was known for his wisdom and wit on the show, which featured him and his family evaluating and purchasing rare and unusual items. Harrisons legacy will be remembered through the show and the impact he had on his family and the community.]'
        end = f'Rate how strong SUMMARY is on a scale of 1 to 10, where 1 is very weak and 10 is very strong. Your rating should focus particularly on the conciseness of the summary: it should include only critical points. Deduct points for any content that could be omitted without significantly altering the summarys meaning. Deduct points if sentences could be shortened, and if unecessary details are included. Deduct points if the summary has logical gaps, or if it appears as though information has been ommitted. Respond only with a number from 1 to 10. '
        prompt = f'{start}\nSUMMARY\n{summary}\n{end}'
    
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        scores = self.run_gpt(messages, model, max_tokens = 1000, temperature = 0)

        return (literal_eval(scores))

    def pipeline(self):
        summarize_prefix = f"Please summarize the following text as succinctly as possible. "  
        
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
        with open("metrics/" + self.name +  "_failures.txt", "w") as f:
            for failure in self.failures:
                f.write(f"{failure}\n")

        with open("metrics/" + self.name + "_baseline_failures.txt", "w") as f:
            for failure in self.baseline_failures:
                f.write(f"{failure}\n")

        plt.style.use('seaborn-deep')
        plt.hist(self.failure_scores, np.linspace(1.0, 10.0, 20), alpha = 0.5, label = 'transfer')
        plt.hist(self.baseline_scores, np.linspace(1.0, 10.0, 20), alpha = 0.5, label = 'baseline')
        plt.legend(loc = 'upper right')
        plt.title("Summarization Score Failure Transfer")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.savefig("metrics/" + self.name + "_histogram.png")
        plt.close()
        
        return (self.name, len(self.failures) / len(self.initial_domain), len(self.baseline_failures) / len(self.baseline))
