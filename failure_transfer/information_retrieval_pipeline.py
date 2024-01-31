from task import Task
import random, math
import re

facts_queries_answers = [
    ("The capital of Japan is Tokyo.", "What is the capital of Japan?", "Tokyo"), 
    ("The chemical symbol for gold is Au.", "What is the chemical symbol for gold?", "Au"), 
    ("The Great Barrier Reef is the largest coral reef system in the world.", "Which coral reef system is the largest in the world?", "The Great Barrier Reef"),
    ("The human body has 206 bones.", "How many bones are in the human body?", "206"), 
    ("Sherlock Holmes is known for residing at 221B Baker Street in London", "What street does Sherlock Holmes live on?", "221B Baker Street"), 
    ("Bobby Fisher has a pet owl named Flappy", "What is the name of Bobby Fischer's owl?", "Flappy"), 
    ("Harry Potter's chessboard has 72 squares arranged in an 8x9 grid.", "How many squares are in Harry Potter's chessboard?", "72") 
]

def get_random_facts(num):
    random.shuffle(facts_queries_answers)
    facts = [tup[0] for tup in facts_queries_answers[:num]]
    queries = [tup[1] for tup in facts_queries_answers[:num]]
    answers = [tup[2] for tup in facts_queries_answers[:num]]
    
    return (facts, queries, answers)

class InformationRetrieval(Task):
    def __init__(self, failure_mode, num_examples, interacter, num_facts = 3):
        super().__init__("information_retrieval_task", failure_mode, num_examples, interacter)
        
        self.num_facts = num_facts
        self.facts, self.queries, self.answers = get_random_facts(num_facts)

    def gen_data(self):
        prompt = f"I will give you a list of facts. Please write me 3 paragraphs, each containing all of the facts in the list. Furthermore, these paragraphs should likely elicit a failure in understanding for the following failure mode. Facts: [" + ", ".join(fact for fact in self.facts) + "Extrapolate based on the failure mode you’ve summarized. Be both creative and cautious. It is critical that each of the 3 paragraphs should every single facts in them."
        
        self.initial_domain = self.gen_failures(prompt)
        self.baseline = self.gen_failures(context = prompt, is_baseline = True)
        
    def pipeline(self):
        info_prefix = f"I will provide you with some text and ask you some questions. Text: "
        info_suffix = f" Questions: "

        for i in range(self.num_facts):
            info_suffix += (str(i + 1) + ". " + self.queries[i] + ", ")

        info_suffix += "Please use the text to answer each of these questions, rather than relying on external knowledge. Do not include the original text in your answer." 
        
        def extract_answers(response):
            return ([response])

        input_domain = self.initial_domain
        print("Scraping Initial Domain...")

        all_failures = []
        for iterations in range(2):
            questions = []
            for i in range(len(input_domain)):
                questions.append(info_prefix + input_domain[i] + info_suffix)

            print(len(questions), questions[0], len(input_domain))
            answers = self.interacter.answer_questions(questions, extract_answers)

            for q in questions:
                print(q)

            print("Answers:")

            for a in answers:
                print(a)

            failures = []
            for i in range(len(answers)):
                pass

            '''
                initial_embedding = self.model.encode(input_domain)
                final_embedding = self.model.encode(answers2[i])
                similarity = util.pytorch_cos_sim(initial_embedding, final_embedding)
    
                print(f"Semantic Similarity Score: {similarity}\n")
            
                if similarity < self.threshold:
                    self.failures.append((input_domain, answer2[i]))
            '''

            all_failures.append(failures)

            print("Scraping Baseline...")
            input_domain = self.baseline

        self.failures = all_failures[0]
        self.baseline_failures = all_failures[1]
