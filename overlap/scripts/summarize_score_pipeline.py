import openai
import numpy as np
import matplotlib.pyplot as plt

import re, random, statistics
from ast import literal_eval

def run_gpt(messages, model, max_tokens = 10, temperature = 0):
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

def score_summary(document, summary, model = 'gpt-3.5-turbo'):
    start = f'You will be given a summary SUMMARY of a document DOCUMENT. Your task will be to assess how strong of a summary SUMMARY is with respect to DOCUMENT.'
    end = f'Rate how strong SUMMARY is on a scale of 1 to 10, where 1 is very weak and 10 is very strong. A strong summary should 1) include all important details from DOCUMENT, 2) have little to no irrelevant or inconsequential details, and 3) retain logical transitions between ideas in DOCUMENT. When scoring SUMMARY, respond only with a number from 1 to 10.'
    prompt = f'{start}\nDOCUMENT\n{document}\nSUMMARY\n{summary}\n{end}'

    messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
    scores = run_gpt(messages, model, max_tokens = 1000, temperature = 0.2)
    
    return (literal_eval(scores))

scores = []

base = 'summarize_failure_output/stories/'
paths = [(base + 'irrelevant_facts.txt', base + 'irrelevant_facts_nofail.txt'), (base + 'sequential_facts.txt', base + 'sequential_facts_nofail.txt'), 
         (base + 'mini_story.txt', base + 'mini_story_nofail.txt'), (base + 'no_change.txt', base + 'no_change_nofail.txt')]

for path, path2 in paths:
    print("Starting path: ", path)

    scraped_failures = []
    scraped_nonfailures = []

    with open(path, 'r') as f:
        lines = f.readlines()
        lines = lines[1:]

        for i in range(0, len(lines), 4):
            document = lines[i].strip().split("Story:")[1]
            summary = lines[i + 1].strip().split("Summary:")[1]
            injected_document = lines[i + 2].strip().split("Story:")[1]
            injected_summary = lines[i + 3].strip().split("Summary:")[1]
            
            scraped_failures.append((document, summary, injected_document, injected_summary))
    
    with open(path2, 'r') as f:
        lines = f.readlines()
        
        for i in range(0, len(lines), 4):
            document = lines[i].strip().split("Story:")[1]
            summary = lines[i + 1].strip().split("Summary:")[1]
            injected_document = lines[i + 2].strip().split("Story:")[1]
            injected_summary = lines[i + 3].strip().split("Summary:")[1]

            scraped_nonfailures.append((document, summary, injected_document, injected_summary)) 
        
    random.shuffle(scraped_failures)
    random.shuffle(scraped_nonfailures)

    failure_scores = []
    failure_datapoints = [] 
    nonfailure_scores = []
    nonfailure_datapoints = []

    for document, summary, injected_document, injected_summary in scraped_failures[:100]:
        cur_score = score_summary(document, summary)

        if type(cur_score) == list:
            failure_scores.append(sum(cur_score))
        else:
            failure_scores.append(cur_score)

        failure_datapoints.append((document, summary, injected_document, injected_summary, cur_score))
    
    for document, summary, injected_document, injected_summary in scraped_nonfailures[:100]:
        cur_score = score_summary(document, summary)

        if type(cur_score) == list:
            nonfailure_scores.append(sum(cur_score))
        else:
            nonfailure_scores.append(cur_score)

        nonfailure_datapoints.append((document, summary, injected_document, injected_summary, cur_score))
    
    failure_datapoints.sort(key = lambda p : p[4])
    nonfailure_datapoints.sort(key = lambda p : p[4]) 

    print(len(failure_scores), len(nonfailure_scores))

    plt.style.use('seaborn-deep')
    plt.hist(failure_scores, np.linspace(1.0, 10.0, 20), alpha = 0.5, label = 'failures')
    plt.hist(nonfailure_scores, np.linspace(1.0, 10.0, 20), alpha = 0.5, label = 'non-failures')
    plt.legend(loc = 'upper right')
    plt.title(path[len(base):-4])
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig(path[:-4] + "_histogram.png")
    plt.close()

    with open(path[:-4] + "_scores.txt", 'w') as f:
        f.write(f"Failures Average score: {statistics.mean(failure_scores)}\n")   
        f.write(f"Failures Standard Deviation: {statistics.pstdev(failure_scores)}\n")
        f.write(f"Non-failures Average score: {statistics.mean(nonfailure_scores)}\n")
        f.write(f"Non-failures Standard Deviation: {statistics.pstdev(nonfailure_scores)}\n")

    with open(path[:-4] + "_failure_ranked_score.txt", 'w') as f:
        for document, summary, injected_document, injected_summary, score in failure_datapoints:
            f.write(f"Original Story: {document}\nOriginal Summary: {summary}\nInjected Story: {injected_document}\nInjected Summary: {injected_summary}\nScore: {score}\n")

    with open(path[:-4] + "_nonfailure_ranked_score.txt", 'w') as f:
        for document, summary, injected_document, injected_summary, score in nonfailure_datapoints:
            f.write(f"Original Story: {document}\nOriginal Summary: {summary}\nInjected Story: {injected_document}\nInjected Summary: {injected_summary}\nScore: {score}\n")
