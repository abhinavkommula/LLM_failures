from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM, GPTQConfig
from sentence_transformers import SentenceTransformer
import transformers

from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import gc

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

    def answer_questions(self, questions, extract_answers, batch_size = 12, max_token_length = 1500):
        questions = [self.messages_to_prompt(self.generate_message(q)) for q in questions]
        answers = []

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
                    answers.extend(extract_answers(response))
                except:
                    answers.append([])

                del output

            inputs = {k : v.cpu() for k, v in inputs.items()}
            del inputs

            torch.cuda.empty_cache()
            gc.collect()

        return (answers)
