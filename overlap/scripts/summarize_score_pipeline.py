import openai
import re, random, statistics

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
    end = f'Rate how accurate SUMMARY is on a scale of 1 to 10, where 1 is not accurate at all and 10 is very accurate. An accurate summary should include all important details, should not hyperfixate on irrelevant or inconsequential details, and retain logical transitions between ideas. Respond only with a number between 1 and 10.'
    prompt = f'{start}\nDOCUMENT\n{document}\nSUMMARY\n{summary}\n{end}'

    messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
    score = run_gpt(messages, model, max_tokens = 1000, temperature = 0.2)
    
    return (int(score))

scores = []

base = 'summarize_failure_output/stories/'
paths = [base + 'irrelevant_facts.txt', base + 'sequential_facts.txt', base + 'mini_story.txt']

for path in paths:
    print("Starting path: ", path)

    scraped_examples = []

    with open(path, 'r') as f:
        lines = f.readlines()
        lines = lines[1:]

        for i in range(0, len(lines), 4):
            print(lines[i], lines[i + 1], lines[i + 2], lines[i + 3])

            document = lines[i].strip().split("Story:")[1]
            summary = lines[i + 1].strip().split("Summary:")[1]
            injected_document = lines[i + 2].strip().split("Story:")[1]
            injected_summary = lines[i + 3].strip().split("Summary:")[1]
            
            scraped_examples.append((document, summary, injected_document, injected_summary))

    random.shuffle(scraped_examples)

    scores = []
    datapoints = [] 
    for document, summary, injected_document, injected_summary in scraped_examples[:100]:
        scores.append(score_summary(document, summary))
        datapoints.append((document, summary, injected_document, injected_summary, scores[-1]))
   
    datapoints.sort(key = lambda p : p[4])

    with open(path[:-4] + "_scores.txt", 'w') as f:
        f.write(f"Average score: {statistics.mean(scores)}\n")   
        f.write(f"Standard Deviation: {statistics.pstdev(scores)}\n")

    with open(path[:-4] + "_ranked_score.txt", 'w') as f:
        for document, summary, injected_document, injected_summary, score in datapoints:
            f.write(f"Original Story: {document}\nOriginal Summary: {summary}\nInjected Story: {injected_document}\nInjected Summary: {injected_summary}\nScore: {score}\n")
