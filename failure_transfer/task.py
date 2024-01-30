from abc import ABCMeta, abstractmethod
import openai
import re

class Task:
    def __init__(self, name, failure_mode, num_examples, interacter):
        self.name = name
        self.failure_mode = failure_mode
        self.num_examples = num_examples
        self.interacter = interacter
    
    def run_gpt(self, messages, model, max_tokens = 10, temperature = 0):
        assert model in ["gpt-4", "gpt-3.5-turbo", 'gpt-4-turbo', 'gpt-3.5-turbo-0613']
    
        if model == 'gpt-4-turbo':
            model = 'gpt-4-1106-preview'
    
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        data = response['choices'][0]['message']['content'].replace('\n\n', '\n')
        return (data)

    def gen_failures(self, context, is_baseline = False, model = 'gpt-3.5-turbo'):
        if is_baseline:
            query = context + '\n'
        else:
            query = context + '\n' + "Failure Mode: [" + self.failure_mode + "]\n"
        
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': query}]
        failures = []
        
        for i in range(int(self.num_examples // 3) + 1):
            llm_output = self.run_gpt(messages, model, max_tokens = 1000, temperature = 0.2)
            paragraphs = re.split(r'\d+:', text)
            failures.extend(paragraphs)

        return (failures)

    def gen_data(self):
        prompt = "Write down 3 short paragraphs that would likely elicit failure in understanding for the following failure mode. You will be evaluated on how well you perform. Your sentence structure and length can be creative; extrapolate based on the failure mode you've summarized. Be both creative and cautious. " 

        baseline_prompt = "Write down 3 short paragraphs. "

        self.intial_domain = self.gen_failures(prompt)
        self.baseline = self.gen_failures(context = baseline_prompt, is_baseline = True)

    ''' 
    Returns: Tuple (Failure Rate, F1 Score, TP Rate, FP Rate, TN Rate, FN Rate)
    '''
    def extract_metrics(self):
        with open("metrics/" + self.name + "_failures.txt", "w") as f:
            for failure in self.failures:
                f.write(f"{failure}\n")

        with open("metrics/" + self.name + "_baseline_failures.txt", "w") as f:
            for failure in self.baseline_failures:
                f.write(f"{failure}\n")

        return (len(self.failures) / self.num_examples, len(self.baseline_failures) / self.num_examples)
