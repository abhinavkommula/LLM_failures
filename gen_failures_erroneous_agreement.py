from tqdm import tqdm
from scrape_jeapordy import JeapordyParser

from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import transformers


from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

'''
Parse json data and extract questions
'''

parser = JeapordyParser()
questions = parser.get_questions()

'''
Pass questions into LLaMA v2 to derive outputs
'''

prefix_q = "Please answer the following question in 1 sentence. Additionally, please write 'The correct answer is:' before your response. "

def generate_message(question):
    formatted_question = question.replace("'", '')

    return ([{'role': 'system', 'content': ''}, 
             {'role': 'user', 'content': (prefix_q + formatted_question)}])

def messages_to_prompt(messages):
    assert len(messages) == 2
    assert messages[0]['role'] == 'system'
    assert messages[1]['role'] == 'user'
    sys_message = f"<s>[INST] <<SYS>>\n{messages[0]['content']}\n<</SYS>>\n\n"
    ins_message= f"{messages[1]['content']} [/INST]"
    prompt = sys_message + ins_message
    return prompt

def load_llama_model():
    model_name_or_path = "/scratch/users/erjones/models/postprocessed_models/7B-chat"
    config = AutoConfig.from_pretrained(model_name_or_path)
    use_fast_tokenizer = "LlamaForCausalLM" not in config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left")
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16,  device_map="auto")
    return model, tokenizer

def extract_answer(llm_response):
    segments = llm_response.replace("[/INST]", "").split("The correct answer is: ")
    return (segments[-1])

print("DEBUG: Passing questions into DistilRoberta to compute answers")

# Work with 1000 examples for now
questions = questions[:2500]
modified_questions = [messages_to_prompt(generate_message(q)) for q in questions]

answers = []
batch_size = 96
idx = 0

model, tokenizer = load_llama_model()
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype = torch.float32, 
    device_map = "auto"
)

for out in tqdm(pipeline(modified_questions, do_sample = True, top_k = 10, num_return_sequences = 1, 
                        eos_token_id = tokenizer.eos_token_id, max_length = 200, batch_size = batch_size)):

    for sequence in out:
        answer = extract_answer(sequence['generated_text'])
        answers.append(answer)
    
        print("Question:", questions[idx], "Answer:", answer)    
        idx += 1

assert(len(questions) == len(answers))

'''
Create embeddings for answers and compare similarities
'''

def load_bert_model():
    bert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    return bert_model

def derive_embeds(data, bert_model):
    embeds = []

    for idx in tqdm(range(0, len(data), batch_size)):
        cur_batch = data[idx : idx + batch_size]

        with torch.no_grad():
            embed_batch = bert_model.encode(cur_batch)

        embed_batch = torch.from_numpy(embed_batch)
        embed_batch = F.normalize(embed_batch, dim = 1)
        
        embeds.append(embed_batch)

    embeds = torch.cat(embeds, dim = 0).cuda()
    return (embeds)


print("DEBUG: Deriving answer embeddings")

bert_model = load_bert_model()
question_embeds = derive_embeds(questions, bert_model)
answer_embeds = derive_embeds(answers, bert_model)

print("DEBUG: Computing similarity between answer embeddings")

non_similar_pairs = []
similar_pairs = []
similarity_threshold = 0.7

# TODO: This is a pretty dumb implementation of similarity comparison, will 
# probably have to implement batching in the future and comparisons within batches

for idx in tqdm(range(0, len(questions), batch_size)):
    question_batch = questions[idx : idx + batch_size]
    answer_batch = answers[idx : idx + batch_size] 

    question_embeds_batch = question_embeds[idx : idx + batch_size]
    answer_embeds_batch = answer_embeds[idx : idx + batch_size]

    question_similarity_matrix = torch.matmul(question_embeds_batch, question_embeds.t())
    answer_similarity_matrix = torch.matmul(answer_embeds_batch, answer_embeds.t())

    mask = (answer_similarity_matrix > similarity_threshold) & (abs(answer_similarity_matrix - question_similarity_matrix) > 0.2)
    indices_i, indices_j = mask.nonzero(as_tuple = True)
    indices_k, indices_l = (mask == 0).nonzero(as_tuple = True)

    for i, j in zip(indices_i.tolist(), indices_j.tolist()):
        question_similarity_score = question_similarity_matrix[i, j].item()
        answer_similarity_score = answer_similarity_matrix[i, j].item()

        similar_pairs.append(((question_batch[i], questions[j]), (answer_batch[i], answers[j]), (question_similarity_score, answer_similarity_score)))

    for k, l in zip(indices_k.tolist(), indices_l.tolist()):
        question_similarity_score = question_similarity_matrix[k, l].item()
        answer_similarity_score = answer_similarity_matrix[k, l].item()

        non_similar_pairs.append(((question_batch[k], questions[l]), (answer_batch[k], answers[l]), (question_similarity_score, answer_similarity_score)))

# Sanity

print("DEBUG: SIMILAR_PAIRS", len(similar_pairs))
for i in range(len(similar_pairs)):
    print(similar_pairs[i])

print("DEBUG: NON_SIMILAR_PAIRS", len(non_similar_pairs))

'''
Prompt GPT to identify patterns in similar pairs
'''

prompt_prefix = """
I will provide a series of data for you to remember. Subsequently, I will ask you some
questions to test your performance! Here are some pairs of prompts for you to memorize."""

prompt_suffix = """
I'm trying to find failures with an embedding model. The above are some pairs of
sentences that it encodes very similarly, even though they're conveying different concepts.
Using these specific examples, are there any general types of failures you notice the
embedding is making, or any common features that the embedding fails to encode? Try
to give failures that are specific enough that someone could reliably produce examples
that the embedding would encode similarly, even though it shouldn't. Please try to give as
many general failures as possible. Please focus on differences that are important visually,
as these embeddings are later used to generate images, or videos. In your failure modes,
please explain clearly why the failure would lead to problems for future tasks related to
visual generation.Please summarize as many as you can and stick to the examples.
"""

full_prompt = prompt_prefix + "["

for pair in similar_pairs:
    full_prompt += "(" + pair[0][0] + "," + pair[0][1] + "), "

full_prompt += ("]. " + prompt_suffix)

print("DEBUG: Resultant prompt ->", full_prompt)

sequence = pipeline(
    messages_to_prompt(generate_message(full_prompt)), 
    do_sample = True, 
    top_k = 10, 
    num_return_sequences = 1, 
    eos_token_id = tokenizer.eos_token_id, 
    max_length = 10000,
    max_new_tokens = 10000
)

response = sequence[0]['generated_text']

print("DEBUG: List of systemic failures", response)


'''
Generate new instances for each systemic failure
'''