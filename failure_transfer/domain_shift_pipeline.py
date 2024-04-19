import numpy as np
import random, math, statistics
import spacy
import re
import os

from task import Task

class DomainShift(Task):
    def __init__(self, failure_mode, num_examples, interacter, logger, initial_domain = [], name = "domain_shift", input_domain = "short_stories", output_domain = "medical_documents", domain_prefix = "", domain_suffix = "", read_file = None):
        super().__init__(name, failure_mode, num_examples, interacter, logger, initial_domain, read_file)
        
        self.input_domain = input_domain
        self.output_domain = output_domain
        self.domain_shift_prefix = domain_prefix
        self.domain_shift_suffix = domain_suffix

    def gen_data(self):
        if len(self.initial_domain) == 0:
            self.logger.print("Num examples: " + str(self.num_examples))
            self.initial_domain = self.scraper.get_examples(self.num_examples)
            self.logger.print(str(len(self.initial_domain)))
            self.baseline = ["..."]
             
        else:
            self.baseline = ["..."] 

    def domain_shift(self, prompt, model = 'gpt-4-turbo'):
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
        output_domain = self.run_gpt(messages, model, max_tokens = 1500, temperature = 0.3)
        output_domain.replace('\n', '; ')

        return (output_domain)     

    def pipeline(self):
        self.results = []
        
        def extract_output(output):
            if "Output:" in output:
                return (':'.join(output.split(":")[1:]))
            else:
                return output
        
        idx = 0
        for example in self.initial_domain:
            prompt = self.domain_shift_prefix + example + self.domain_shift_suffix
            self.results.append(extract_output(self.domain_shift(prompt)))
            
            if idx % 10 == 0:
                self.logger.print("BATCH: " + str(idx))
        
            idx += 1

    def extract_metrics(self):
        with open("data/domain_shift/" + self.input_domain.replace(' ', '_') + "_" + self.output_domain.replace(' ', '_') + "_examples.txt", "a+") as f:
            for res in self.results:
                f.write(f"{res}\n")
            
        return (self.name, -1, -1)
