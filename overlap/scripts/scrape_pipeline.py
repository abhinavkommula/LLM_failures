from scrape_tinystories import TinyStoriesParser
from scrape_news import NewsParser
from scrape_file import FileParser
from scrape_hf_dataset import HFParser
import random, math
import re
import gc
import os

#parser = NewsParser()
parser = TinyStoriesParser()
#parser = FileParser('../data/failure_test_transfer/physical_mental_news_random.txt', lambda line : line.replace('"', '').strip())

'''
def poetry_parser(x):
    poem = x.strip()

    if len(poem.split(' ')) > 200:
        poem = ""

    return (poem)

parser = HFParser("merve/poetry", poetry_parser)
'''

stories = parser.get_stories()

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM, GPTQConfig
from sentence_transformers import SentenceTransformer
import transformers

from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class InteractLLaMA:
    def __init__(self):
        self.model, self.tokenizer = self.load_llama_model()
            
    def generate_message(self, question):
        formatted_question = question.replace("'", '')
        return ([{'role': 'system', 'content': ''}, 
                 {'role': 'user', 'content': (formatted_question)}])

    def messages_to_prompt(self, messages):
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
        sys_message = f"<s>[INST] <<SYS>>\n{messages[0]['content']}\n<</SYS>>\n\n"
        ins_message= f"{messages[1]['content']} [/INST]"
        prompt = sys_message + ins_message
        return prompt

    def load_llama_model(self):
        model_name_or_path = "/scratch/users/erjones/models/postprocessed_models/7B-chat"
        config = AutoConfig.from_pretrained(model_name_or_path)
        use_fast_tokenizer = "LlamaForCausalLM" not in config.architectures
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left")
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
        
        #gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer = tokenizer)      
        #model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=gptq_config, device_map="auto")
        
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
        return model, tokenizer

    def answer_questions(self, questions):
        questions = [self.messages_to_prompt(self.generate_message(q)) for q in questions]
        answers = []

        batch_size = 12
        max_token_length = 1500
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for idx in range(0, len(questions), batch_size):
            batch = questions[idx : min(len(questions), idx + batch_size)]
            print("BATCH:", len(batch))

            inputs = self.tokenizer(batch, return_tensors = 'pt', padding = True, max_length = max_token_length)
            inputs = {k : v.to(device) for k, v in inputs.items()}

            token_counts = [input_id.size(0) for input_id in inputs["input_ids"]]

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length = max_token_length, num_return_sequences = 1)

            for output, count in zip(outputs, token_counts):
                response = self.tokenizer.decode(output[count:], skip_special_tokens = True)
                response = response.replace('\r', '').replace('\n', '')
                response = response.split(":")[-1]

                try:
                    items = re.split(r'\d+\.\s*', response)
                    truncate_to_two_words = lambda x : ' '.join(x.split(' ')[:5])
                    answers.append([truncate_to_two_words(item.strip()) for item in items if item])

                except:
                    answers.append([])

                del output

            inputs = {k : v.cpu() for k, v in inputs.items()}
            del inputs

            torch.cuda.empty_cache()
            gc.collect()

        return (answers)

'''
Verify overlap amongst pairs of opposite entites in dataset
'''

#list_1 = ["protagonists", "physical actions"]
#list_2 = ["antagonists", "mental actions"]

list_1 = ["protagonists", "indoor settings", "physical actions", "male entities", "positive emotions"]
list_2 = ["antagonists", "outdoor settings", "mental actions", "female entities", "negative emotions"]

'''
list_1 = ["protagonists", "old characters", "male characters", "indoor settings", "physical actions", "human characters", "positive emotions"]
list_2 = ["antagonists", "young characters", "female characters", "outdoor settings", "mental actions", "animal characters", "negative emotions"]
'''

def extract_questions(list_of_stories, search_text):
    prefix_q = "I will provide you with some sample text. From this text, please provide a list of all examples of " + search_text + ". Limit each answer in the list to 1 to 2 words. When writing your answer, generate it as: '[List: 1., 2., 3., ...]'"

    new_questions = []
    for story in list_of_stories:
        story = story.strip()
        new_questions.append(prefix_q + story)

    return (new_questions)
    
import random
random.shuffle(stories)

stories = list(filter(lambda x : x != "" and len(x.split(' ')) <= 60, stories))
short_stories = stories[:1000]
print("Number of possible stories: ", len(stories))

interacter = InteractLLaMA()

# Indexed by failure-test pair
overlaps = {}

total_errors = [0 for el in list_1]
total_questions = [0 for el in list_1]

for i in range(len(list_1)):
    overlaps[list_1[i]] = []

    questions_1 = extract_questions(short_stories, list_1[i])
    answers_1   = interacter.answer_questions(questions_1) 

    questions_2 = extract_questions(short_stories, list_2[i])
    answers_2   = interacter.answer_questions(questions_2)

    for a_idx in range(len(answers_1)):
        set1 = set(answers_1[a_idx])
        set2 = set(answers_2[a_idx])
        overlap = set1.intersection(set2)

        if overlap:
            total_errors[i] += 1 

            overlap_ratio = len(overlap) / (len(set1) + len(set2) - len(overlap))
            overlaps[list_1[i]].append((list_1[i], list_2[i], answers_1[a_idx], answers_2[a_idx], short_stories[a_idx], overlap_ratio))

        total_questions[i] += 1

    overlaps[list_1[i]].sort(key = lambda tup : -tup[5])

'''
Output file structure:

{language domain}_{domain type}_{failure test}_{?random}
    /all_overlaps.txt
    /classification_prompt.txt
    /statistics.txt
'''

output_directory = 'scrape_output/news_indomain_all'

try:
    os.mkdir(output_directory)
except OSError as error:
    print(f"Could not create directory along path {output_directory}; {error}\n")


with open(output_directory + '/statistics.txt', 'w') as f:
    for i in range(len(list_1)):
        error_rate = (total_errors[i] * 1.0) / total_questions[i]
        binomial_error = 1.96 * math.sqrt((error_rate * (1 - error_rate)) / total_questions[i])

        f.write(f"{list_1[i]}/{list_2[i]} Error rate: {error_rate}, Confidence interval: [{error_rate - binomial_error}, {error_rate + binomial_error}]\n")


num_examples_per_failure_test = 20
with open(output_directory + '/classification_prompt.txt', 'w') as f:
    f.write("[")

    for failure_test in list_1:
        for o in overlaps[failure_test][:num_examples_per_failure_test]:
            o = o.replace("\n", "")
            f.write(f"({o[4]})\n")

    f.write("]")

with open(output_directory + '/all_overlaps.txt', 'w') as f:
    for failure_test in list_1:
        for o in overlaps[failure_test]:
            try:
                f.write(f"Story: {o[4]}\nEntity {o[0]}: {o[2]}\nEntity {o[1]}: {o[3]}\nOverlap Ratio: {o[5]}\n")
            except:
                continue
