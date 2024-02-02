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
   
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            data = response['choices'][0]['message']['content'].replace('\n\n', '\n')
        except:
            data = "Error Querying OpenAI"

        return (data)

    def gen_failures(self, context, num_paragraphs = 5, is_baseline = False, model = 'gpt-3.5-turbo'):
        iteration_number = self.num_examples

        if is_baseline:
            query = context + "Only write your attack paragraphs. Do not include any text about a language model. "
            iteration_number = (iteration_number // 2)
        else:
            query = context + "Failure Mode: [" + self.failure_mode + "]. Only write your attack paragraphs. Do not include any language model. "

        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': query}]
        failures = []
        
        for i in range(int(iteration_number // num_paragraphs)):
            llm_output = self.run_gpt(messages, model, max_tokens = 1000, temperature = 0.3).replace("Attack", '')
            paragraphs = re.split(r'\d+.', llm_output)[1:]
           
            #print(llm_output, len(paragraphs))
            if len(paragraphs) != num_paragraphs:
                continue

            failures.extend(paragraphs)
            print("Failures Generated:", num_paragraphs * (i + 1))

        return (failures)

    def gen_data(self):
        raise NotImplementedError
    
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

        return (self.name, len(self.failures) / len(self.initial_domain), len(self.baseline_failures) / len(self.baseline))
