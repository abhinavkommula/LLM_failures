import openai
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

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
    start = f'You will be given a summary SUMMARY, and your task will be to assess how strong of a summary it is on scale from 1 to 10. Here is an example of a summary that would recieve a score of 2, because a high proportion of it contains irrelevant text: [a $4 trillion package with revenues, a $1 trillion-$2 trillion spending cuts-only deal, or a deal that grants the President authority to raise the debt ceiling without GOP agreement. Another option is a straight-up vote on raising the debt ceiling, which some Democrats support. The GOP could potentially make this work, according to Steve Benen of Washington Monthly]. Here is an example of a summary that would receive a score of 7 because it has a smaller portion of irrelevant information and makes more logical sense: [Richard "The Old Man" Harrison, a star of the popular reality TV show "Pawn Stars," has passed away at the age of 77 due to complications from Parkinsons disease. He will be greatly missed by his family, the team at Gold & Silver Pawn, and his many fans around the world. Harrison was a Navy veteran who opened the Gold & Silver Pawn Shop in Las Vegas in 1988, which became a multimillion-dollar business after the show premiered in 2009. He was known for his wisdom and wit on the show, which featured him and his family evaluating and purchasing rare and unusual items. Harrisons legacy will be remembered through the show and the impact he had on his family and the community.]'
    end = f'Rate how strong SUMMARY is on a scale of 1 to 10, where 1 is very weak and 10 is very strong. Your rating should focus particularly on the conciseness of the summary: it should include only critical points. Deduct points for any content that could be omitted without significantly altering the summarys meaning. Deduct points if sentences could be shortened, and if unecessary details are included. Deduct points if the summary has logical gaps, or if it appears as though information has been ommitted. Respond only with a number from 1 to 10. '
    prompt = f'{start}\nSUMMARY\n{summary}\n{end}'
    
    messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
    scores = run_gpt(messages, model, max_tokens = 1000, temperature = 0)
    print(scores)

    return (literal_eval(scores))

scores = []

base = 'summarize_failure_output/news/'
paths = [(base + 'irrelevant_facts.txt', base + 'irrelevant_facts_nofail.txt')] #(base + 'sequential_facts.txt', base + 'sequential_facts_nofail.txt'), 
#         (base + 'mini_story.txt', base + 'mini_story_nofail.txt'), (base + 'no_change.txt', base + 'no_change_nofail.txt')]

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

    print(len(scraped_failures), len(scraped_nonfailures))
    
    print("Scoring Failures:")

    for document, summary, injected_document, injected_summary in scraped_failures[:500]:
        cur_score = score_summary(document, summary)

        if type(cur_score) == list:
            failure_scores.append(sum(cur_score))
        else:
            failure_scores.append(cur_score)

        failure_datapoints.append((document, summary, injected_document, injected_summary, cur_score))
        print(len(failure_datapoints))

    print("Scoring Nonfailures:")
    
    for document, summary, injected_document, injected_summary in scraped_nonfailures[:500]:
        cur_score = score_summary(document, summary)

        if type(cur_score) == list:
            nonfailure_scores.append(sum(cur_score))
        else:
            nonfailure_scores.append(cur_score)

        nonfailure_datapoints.append((document, summary, injected_document, injected_summary, cur_score))
        print(len(nonfailure_datapoints))

    print(len(failure_datapoints), len(nonfailure_datapoints))
    
    failure_datapoints.sort(key = lambda p : p[4])
    nonfailure_datapoints.sort(key = lambda p : p[4]) 

    if len(failure_scores) == 0:
        failure_scores.append(1)
    if len(nonfailure_scores) == 0:
        nonfailure_scores.append(1)

    plt.style.use('seaborn-deep')
    plt.hist(failure_scores, np.linspace(1.0, 10.0, 20), alpha = 0.5, label = 'failures')
    plt.hist(nonfailure_scores, np.linspace(1.0, 10.0, 20), alpha = 0.5, label = 'non-failures')
    plt.legend(loc = 'upper right')
    plt.title(path[len(base):-4])
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig(path[:-4] + "_histogram.png")
    plt.close()

    failure_scores_np = np.array(failure_scores)
    nonfailure_scores_np = np.array(nonfailure_scores)

    with open(path[:-4] + "_failure_ranked_score.txt", 'w') as f:
        for document, summary, injected_document, injected_summary, score in failure_datapoints:
            f.write(f"Original Story: {document}\nOriginal Summary: {summary}\nInjected Story: {injected_document}\nInjected Summary: {injected_summary}\nScore: {score}\n")

    with open(path[:-4] + "_nonfailure_ranked_score.txt", 'w') as f:
        for document, summary, injected_document, injected_summary, score in nonfailure_datapoints:
            f.write(f"Original Story: {document}\nOriginal Summary: {summary}\nInjected Story: {injected_document}\nInjected Summary: {injected_summary}\nScore: {score}\n")
    
    emd = wasserstein_distance(failure_scores, nonfailure_scores)
    #b_distance = np.sum(np.sqrt((failure_scores_np / np.sum(failure_scores_np)) * (nonfailure_scores_np / np.sum(nonfailure_scores_np))))

    with open(path[:-4] + "_scores.txt", 'w') as f:
        f.write(f"Failures Average score: {statistics.mean(failure_scores)}\n")   
        f.write(f"Failures Standard Deviation: {statistics.pstdev(failure_scores)}\n")
        f.write(f"Non-failures Average score: {statistics.mean(nonfailure_scores)}\n")
        f.write(f"Non-failures Standard Deviation: {statistics.pstdev(nonfailure_scores)}\n")
        f.write(f"Earth Movers Distance: {emd}\n")
        f.write(f"Bhattacharyya Coefficient: {b_distance}\n")
