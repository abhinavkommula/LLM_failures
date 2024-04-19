from task import Task
import random, math
import re

class InformationRetrieval(Task):
    def __init__(self, failure_mode, num_examples, interacter, num_facts = 3):
        super().__init__("information_retrieval_task", failure_mode, num_examples, interacter)
        
        self.num_facts = num_facts

    def gen_data(self):
        prompt = f"Write 3 additional paragraphs, and for each paragraph provide {self.num_facts} objective question/short answer pairs that could be used to test a language models ability to retrieve information. Furthermore, design all 3 of these paragraphs so that when passed as input to a language model with the following failure mode they would likely elicit failures in information retrieval tasks. You will be evaluated on how well you perform. Your sentence structure and length can be creative; extrapolate based on the failure mode. Be both creative and cautious. Format your output as: (Paragraph 1., Questions ...), (Paragraph 2., Questions ...). "
        baseline_prompt = f"Write 3 additional paragraphs, and for each paragraph provide {self.num_facts} objective question/short answer pairs that could be used to test a language models ability to retrieve information. Furthermore, design all 3 of these paragraphs so that when passed as input to a language model they would likely elicit failures in information retrieval tasks. You will be evaluated on how well you perform. Your sentence structure and length can be creative; be both creative and cautious. "

        self.initial_domain = self.gen_failures(context = prompt, num_paragraphs = 3)
        self.baseline = self.gen_failures(context = baseline_prompt, num_paragraphs = 3, is_baseline = True)

    def pipeline(self):
        info_prefix = f"I will provide you with a paragraph and {self.num_facts} questions related to this paragraph. Please answer them to the best of your ability, and rely only on the paragraph. Paragraph: "

        def extract_question_answer(example):
            split_qs = example.replace('\n', '').split("1.")
            questions_indiv = re.split(r'\d+.', split_qs[1])

            questions = []
            answers = []

            for q in questions_indiv:
                if '-' in q:
                    divide = q.split('-')
                    questions.append(divide[0])
                    answers.append(divide[1])
                else:
                    questions.append('0')
                    answers.append('0')

            return (split_qs[0], questions, answers)

        def extract_answers(response):
            return ([response])

        input_domain = self.initial_domain
        all_failures = []

        for iterations in range(2):
            questions = []
            question_answers = []

            for i in range(len(input_domain)):
                orig_q, extract_qs, extract_ans = extract_question_answer(input_domain[i])
                
                questions_str = ""
                for i in range(len(extract_qs)):
                    questions_str += (str(i + 1) + ".")
                    questions_str += (extract_qs[i] + " ")

                questions.append(info_prefix + orig_q + " Questions: " + questions_str)
                question_answers.append((extract_qs, extract_ans))

            answers = self.interacter.answer_questions(questions, extract_answers)

            failures = []
            for i in range(len(answers)):
                print("Questions:", questions[i], "Answers:", answers[i], question_answers[i])                 
            
            input_domain = self.baseline
            all_failures.append(failures)

        self.failures = all_failures[0]
        self.baseline_failures = all_failures[1]
