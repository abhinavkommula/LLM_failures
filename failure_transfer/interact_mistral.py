from tqdm import tqdm

from transformers import Conversation, AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM, GPTQConfig
from sentence_transformers import SentenceTransformer
import transformers

from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import gc

from model_utils import query_mistral_vllm as query_mistral
from model_utils import load_mistral_model_vllm as load_mistral_model

class InteractMistral:
    def __init__(self):
        pass

    def __init__(self, logger):
        self.logger = logger
        self.model_name = 'mistral-7B-instruct-v0.2'
        self.model = load_mistral_model(self.model_name)

    def answer_questions(self, questions, extract_answer):
        modified_questions = []
        for question in questions:
            modified_questions.append([{'role': 'system', 'content': 'Respond to queries exactly, without additional explanation.'}, {'role': 'user', 'content': question}])
        
        answers = query_mistral(modified_questions, self.model, self.model_name, max_tokens = 3000, temperature = 0.3)
        extracted_answers = []
        
        for answer in answers:
            extracted_answers.extend(extract_answer(answer))
        
        return (extracted_answers)
    
'''
class InteractMistral:
    def __init__(self):
        pass
    
    def __init__(self, logger):
        gc.collect()
        torch.cuda.empty_cache()
        
        self.load_mistral_model()
        self.logger = logger

    def messages_to_prompt(self, question):        
        formatted_question = question.replace("'", '')
        return (self.tokenizer.apply_chat_template([{"role": "user", "content": formatted_question}], tokenize = False))

    def load_mistral_model(self):        
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def answer_questions(self, questions, extract_answers, batch_size = 10, max_token_length = 1500):           
        questions = [self.messages_to_prompt(q) for q in questions]
        answers = []
    
        gc.collect()
        torch.cuda.empty_cache()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        for idx in range(0, len(questions), batch_size):
            batch = questions[idx : min(len(questions), idx + batch_size)]            
            self.logger.print("BATCH: " + str(len(batch)))

            try:
                max_token_length = 1500
                inputs = self.tokenizer(batch, return_tensors='pt', padding=True, max_length=max_token_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_length=max_token_length, num_return_sequences=1)
                
                for output in outputs:    
                    response = self.tokenizer.decode(output, skip_special_tokens = True)
                    response = "".join(response.replace('\r', '').replace('\n', '').split("[/INST]")[1:])
                
                    answers.extend(extract_answers(response))

                    del output
                    torch.cuda.empty_cache()
            
                inputs = {k: v.cpu().detach() for k, v in inputs.items()}
                del inputs
                
            except ValueError:
                self.logger.print("Exceeded max length of 3000, retrying with max length of 5000 and smaller batch size.")
                smaller_batch_size = 3
                
                for sub_idx in range(0, len(batch), smaller_batch_size):
                    sub_batch = batch[sub_idx : min(len(batch), sub_idx + smaller_batch_size)]
                    self.logger.print("SUB BATCH: " + str(len(sub_batch)))

                    max_token_length = 5000
                    inputs = self.tokenizer(sub_batch, return_tensors='pt', padding=True, max_length=max_token_length)
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model.generate(**inputs, max_length=max_token_length, num_return_sequences=1)  
        
                    for output in outputs:    
                        response = self.tokenizer.decode(output, skip_special_tokens = True)
                        response = "".join(response.replace('\r', '').replace('\n', '').split("[/INST]")[1:])
                    
                        answers.extend(extract_answers(response))

                        del output
                        torch.cuda.empty_cache()
                    
                    inputs = {k: v.cpu().detach() for k, v in inputs.items()}
                    del inputs

            torch.cuda.empty_cache()
            gc.collect()
        
        return (answers)
'''