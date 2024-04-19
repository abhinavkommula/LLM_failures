import os
import random

from ast import literal_eval
from sentence_transformers import SentenceTransformer, util

def avg(lst):
    return sum(lst) / len(lst)

languages = ["spanish", "french", "dutch"]

model = SentenceTransformer('paraphrase-distilroberta-base-v1')
cur_dir = './data/'
prompt_examples = 200

pairs = []
total_pairs = []
mp = {}
mp2 = {}

failure_mode_prefix = "I will provide a series of data for you to remember. Subsequently, I will ask you some questions to test your performance! Here are some prompts for you to memorize."

failure_mode_suffix = "I am looking for failures with a language model. Below are pairs of sentences that are written differently, even though they convey similar concepts. The first sentence in each pair is a sentence that a language model translates poorly, while the second sentence is an example it translates well. Using these specific examples, are there any common differences you notice between weak and strong pairs? Focus on differences you observe between the weak and strong sentences that would hinder a language model's ability to translate text accurately. Try to give features that are specific enough that someone could reliably produce examples that the language model would translate poorly. In your failure modes, explain clearly why the difference between weak and strong sentence would lead to problems for future tasks related to translation. Focus on the most common differences across weak and strong sentences."

failure_mode_suffix_2 = "I am looking for failures with a language model. Below are pairs of sentences, the original sentence along with the back-translation performed by the languge model (the result of translating the original from english to Spanish, and then from Spanish back to english). The back-translated exmaples are not strong translations, as it differs significantly in meaning and structure from the original sentence. Using these specific examples, are there any common features you notice in the original sentences that lead to differences in the back-translated sentences? Try to give features that are specific enough that someone could reliably produce examples that the language model would translate poorly. In your failure modes, explain clearly why the feature in the original sentences would lead to downstream failures in back-translated examples. Focus on the most common features across original sentences that lead to differences."

failure_mode_suffix_3 = "I am looking for failures with a language model. Below are pairs of sentences, the original sentence along with the back-translation performed by the languge model (the result of translating the original from english to Spanish, and then from Spanish back to english). The back-translated exmaples are not strong translations, as it differs significantly in meaning and structure from the original sentence. Using these specific examples, are there any common failures you notice amongst the back-translated sentences and the original sentences? Try to give differences in the back-translated sentences that are specific, along with a clear description of the differences you notice with examples. Focus on the most common features across back-translated sentences."

failure_mode_suffix_4 = "I am looking for failures with a language model. Below are weak sentences that a language model translates poorly, along with strong sentences that a language model translates well. Using these specific examples, are there any common differences you notice between weak and strong sentences? Focus on differences you observe between the weak and strong sentences that would hinder a language model's ability to translate text accurately. Try to give features that are specific enough that someone could reliably produce examples that the language model would translate poorly. In your failure modes, explain clearly why the difference between weak and strong sentence would lead to problems for future tasks related to translation. Focus on the most common differences across weak and strong sentences."

failure_mode_suffix_5 = "I am looking for failures with a language model. Below are pairs of sentences that are written differently, even though they convey similar concepts. Using these specific examples, are there any common differences you notice between weak and strong pairs? Focus on differences in the structure of the text between each weak & strong pair, and be specific in the differences you notice so that someone could reliably reproduce these differences in textual structure. Focus on the most common differences, and structures that are important for translation as these differences will be used to assess a language model's ability to translate text."

for filename in os.listdir(cur_dir):
    file_path = os.path.join(cur_dir, filename)

    if os.path.isfile(file_path):
        words = filename.split('_')
        
        if "baseline" not in file_path and "translation_failure_mode" in file_path and  words[-1] == 'failures.txt':
            if words[3] == "paired":
                failure_mode_id = words[4]
                aggregate = True
            else:
                failure_mode_id = words[3]
            
            weak_pairs_analysis = []
            
            vis = {}
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    tup = literal_eval(line)

                    if tup[0] in vis:
                        continue
                    elif "I am sorry" in tup[1] or "I'd be happy to help" in tup[1] or len(tup[1]) < 5:
                        continue
                    
                    vis[tup[0]] = True
                    
                    if tup[3] == 0:
                        if not aggregate:
                            weak_pairs_analysis.append((tup[0].replace('\n', ''), tup[1].replace('\n', '')))
                        else:
                            total_pairs.append((tup[0].replace('\n', ''), tup[1].replace('\n', ''), tup[2]))

            with open(f"prompts/translation_failure_mode_{failure_mode_id}_analysis_prompt.txt", "w") as f:
                f.write(f"{failure_mode_prefix}\n[")
    
                for pair in weak_pairs_analysis[:prompt_examples]:
                    f.write(f"(Original: {pair[0]}\nBack-translated: {pair[1]}),\n")
    
                f.write(f"]\n{failure_mode_suffix_3}\n")
        
        #elif len(words) == 3 and words[0].lower() in languages and words[-1] == 'failures.txt' and words[1] == "adaptive":
        elif len(words) == 2 and words[0].lower() in languages and words[1] == 'failures.txt':
            print("Examining file_path:", file_path)
            
            vis = {}
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    tup = literal_eval(line)

                    if tup[0] in vis:
                        continue
                    elif "I am sorry" in tup[1] or "I'd be happy to help" in tup[1] or len(tup[1]) < 5:
                        continue
                    
                    vis[tup[0]] = True
                    sentence = tup[0].replace('\n', '')
                    
                    if sentence not in mp:
                        mp[sentence] = []
                        mp2[sentence] = []

                    mp[sentence].append(tup[2])
                    mp2[sentence].append(tup[3])
                    pairs.append((tup[0].replace('\n', ''), tup[1].replace('\n', ''), ))


total_pairs.sort(key = lambda t: t[2])
with open(f"prompts/translation_failure_mode_paired_rompt.txt", "w") as f:
    f.write(f"{failure_mode_prefix}\n[")
    
    for pair in total_pairs[:prompt_examples]:
        f.write(f"(Original: {pair[0]}\nBack-translated: {pair[1]}),\n")
    
    f.write(f"]\n{failure_mode_suffix_3}\n")

failures = []
nonfailures = []
weak_pairs = []
strong_pairs = []
used = {}

for sentence in mp2:
    print(sentence, mp2[sentence], mp[sentence])
    
    if sum(mp2[sentence]) >= 2 and sentence not in used:
        nonfailures.append(sentence)
        used[sentence] = True
    elif sum(mp2[sentence]) <= 1 and sentence not in used:
        failures.append(sentence)
        used[sentence] = True

used = {}
for p in pairs:
    sentence = p[0]
    
    if sum(mp2[sentence]) <= 1 and sentence not in used:
        weak_pairs.append(p)
        used[sentence] = True
    elif sum(mp2[sentence]) >= 3 and sentence not in used:
        strong_pairs.append(p)
        used[sentence] = True
        
failures.sort(key = lambda t: max(mp[t]))
weak_pairs.sort(key = lambda t: (sum(mp2[t[0]]), -len(t[0])))
strong_pairs.sort(key = lambda t: (sum(mp2[t[0]]), -len(t[0])))

'''
with open("prompts/translation_failure_prompt.txt", "w") as f:
    f.write(f"{failure_mode_prefix}\nWeak Examples: [\n")
    
/accounts/projects/jsteinhardt/akommula/LLM_failures/.ipynb_checkpoints    for pair in weak_pairs[:prompt_examples//2]:
        f.write(f"({pair[0]}),\n")
        #f.write(f"(Original: {pair[0]}\nBack-translated: {pair[1]}),\n")
    
    f.write(f"]\nStrong Examples: [\n")

    for pair in strong_pairs[-prompt_examples//2:]:
        f.write(f"({pair[0]}),\n")

    f.write(f"]\n{failure_mode_suffix_4}\n")

'''
embeddings_failure = model.encode(failures, convert_to_tensor = True)
embeddings_nonfailure = model.encode(nonfailures, convert_to_tensor = True)

failure_examples = []
baseline_examples = []
            
for i, embedding_A in enumerate(embeddings_failure):
    similarities = util.pytorch_cos_sim(embedding_A.unsqueeze(0), embeddings_nonfailure)
    similarity_scores = similarities.flatten().tolist()
    closest_index = similarities.argmax()
    
    if similarity_scores[closest_index] >= 0.5:
        failure_examples.append((failures[i], nonfailures[closest_index], similarity_scores[closest_index])) 
    
failure_examples.sort(key = lambda tup: -tup[2])

with open("prompts/translation_failure_prompt.txt", "w") as f:
    f.write(f"{failure_mode_prefix}\n[")
    
    for example in failure_examples[:prompt_examples]:
        f.write(f"(Weak: {example[0]}\nStrong: {example[1]}),\n")
    
    f.write(f"]\n{failure_mode_suffix_5}\n")
        
with open("prompts/translation_baseline_prompt.txt", "w") as f:
    f.write(f"{failure_mode_prefix}\n[")

    for example in baseline_examples:
        f.write(f"({example}),\n")
                    
    f.write(f"]\n{failure_mode_suffix}\n")
