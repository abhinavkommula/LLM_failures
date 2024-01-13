from scrape_tinystories import TinyStoriesParser
from scrape_generated_tinystories import AdaptiveStoriesParser
from scrape_news import NewsParser
from scrape_file import FileParser
from scrape_hf_dataset import HFParser
import random
import re
import gc

#parser = NewsParser()
#parser = AdaptiveStoriesParser('../data/adaptive_stories_protagonist_antagonist.txt')
#parser = TinyStoriesParser()
#parser = FileParser('../data/news_to_news.txt', lambda line : line.replace('"', '').strip())

def poetry_parser(x):
    poem = x.strip()

    if len(poem.split(' ')) > 200:
        poem = ""

    return (poem)

parser = HFParser("merve/poetry", poetry_parser)

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
        new_questions.append(prefix_q + story)

    return (new_questions)
    
import random
random.shuffle(stories)
short_stories = stories[:500]

interacter = InteractLLaMA()
overlaps = []

total_errors = [0 for el in list_1]
total_questions = [0 for el in list_1]

for i in range(len(list_1)):
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
            overlaps.append((list_1[i], list_2[i], answers_1[a_idx], answers_2[a_idx], questions_1[a_idx]))

        total_questions[i] += 1

    print(questions_1)
    print(answers_1)
    print(answers_2)

#output_filename = 'baseline_tinystory_overlaps_indoor_outdoor.txt'
#output_filename = 'adaptive_tinystory_overlaps_protaginist_antagonist.txt'
#output_filename = 'tinystory_overlaps_physical_mental.txt'
#output_filename = 'news_overlaps_all.txt'
#output_filename = 'news_to_news_overlaps_all.txt'
output_filename = 'poetry_overlaps_all.txt'

with open('output/' + output_filename, 'w') as f:
    for i in range(len(list_1)):
        f.write(f"Error rate pair {list_1[i]}/{list_2[i]}: {(total_errors[i] * 1.0) / total_questions[i]}\n")

    for o in overlaps:
        try:
            f.write(f"Story: {o[4]}\nEntity {o[0]}: {o[2]}\nEntity {o[1]}: {o[3]}\n\n")
        except:
            continue
