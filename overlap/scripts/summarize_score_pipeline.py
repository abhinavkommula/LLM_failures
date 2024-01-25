import openai
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
    start = f'You will be given a summary SUMMARY of a document DOCUMENT. Your task will be to assess how accurate of a summary SUMMARY is with respect to DOCUMENT.'
    end = f'Rate how accurate SUMMARY is on a scale of 1 to 100, where 1 is not accurate at all and 100 is very accurate. An accurate summary should 1) include all important details from DOCUMENT, 2) should not hyperfixate on irrelevant or inconsequential details, and 3) retain logical transitions between ideas. When scoring SUMMARY, respond only with a list of three numbers which represents the score across each of the three categories (for example, [20, 30, 15]).'
    prompt = f'{start}\nDOCUMENT\n{document}\nSUMMARY\n{summary}\n{end}'

    messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
    scores = run_gpt(messages, model, max_tokens = 1000, temperature = 0.2)
    
    return (literal_eval(scores))

scores = []

base = 'summarize_failure_output/stories/'
paths = [base + 'irrelevant_facts.txt', base + 'sequential_facts.txt', base + 'mini_story.txt', base + 'no_change.txt']

for path in paths:
    print("Starting path: ", path)

    scraped_examples = []

    with open(path, 'r') as f:
        lines = f.readlines()
        lines = lines[1:]

        for i in range(0, len(lines), 4):
            document = lines[i].strip().split("Story:")[1]
            summary = lines[i + 1].strip().split("Summary:")[1]
            injected_document = lines[i + 2].strip().split("Story:")[1]
            injected_summary = lines[i + 3].strip().split("Summary:")[1]
            
            scraped_examples.append((document, summary, injected_document, injected_summary))

    random.shuffle(scraped_examples)

    scores = []
    datapoints = [] 
    for document, summary, injected_document, injected_summary in scraped_examples[:100]:
        cur_score = score_summary(document, summary)

        scores.append(sum(cur_score))
        datapoints.append((document, summary, injected_document, injected_summary, cur_score))
   
    datapoints.sort(key = lambda p : sum(p[4]))

    with open(path[:-4] + "_scores.txt", 'w') as f:
        f.write(f"Average score: {statistics.mean(scores)}\n")   
        f.write(f"Standard Deviation: {statistics.pstdev(scores)}\n")

    with open(path[:-4] + "_ranked_score.txt", 'w') as f:
        for document, summary, injected_document, injected_summary, score in datapoints:
            f.write(f"Original Story: {document}\nOriginal Summary: {summary}\nInjected Story: {injected_document}\nInjected Summary: {injected_summary}\nScores: {score}\nTotal Score: {sum(score)}\n")
