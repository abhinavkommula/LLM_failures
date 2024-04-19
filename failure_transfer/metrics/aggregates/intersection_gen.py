from ast import literal_eval
import itertools
import openai
import random

stories = []

with open("intersections.txt", "r") as f:
    lines = f.readlines()
    
    for i in range(0, len(lines), 2):
        line_0 = lines[i].strip()
        line_1 = lines[i + 1].strip()
        
        if len(line_1) >= 4:
            stories.append(line_0[1:    ])

# Identify all subsets of length 3

def run_gpt(messages, model, max_tokens = 10, temperature = 0):
    assert model in ["gpt-4", "gpt-3.5-turbo", 'gpt-4-turbo', 'gpt-3.5-turbo-0613']

    if model == 'gpt-4-turbo':
        model = 'gpt-4-1106-preview'

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        data = response['choices'][0]['message']['content'].replace('\n\n', '\n')
    except KeyboardInterrupt:
        print("Interruption...")
        exit()
    except Exception:
        data = "Error Querying OpenAI"

    return (data)

def gen_failure_instance(prompt, model = 'gpt-4-turbo'):
    messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]
    output = run_gpt(messages, model, max_tokens = 1500, temperature = 0)
    output = output.replace('\n', '')
    
    return (output)    

prompt_prefix = "Here are three examples of short stories. Pay attention to the structure and contents of these stories, as I will ask you to generate a subsequent story using these three example short stories:\n["
prompt_suffix = "]\nUsing these examples, please write a new short story that uses elements from all three of the above stories. Your output should reasonably match the length, complexity, structure, content-difficulty, etc. of these three example stories provided."
iter = 0

for subset in itertools.combinations(stories, 3):
    subset = list(subset)
    random.shuffle(subset)
    
    print("Iteration:", iter)
    print("Generating using Subset:", subset)

    prompt_middle = ""
    for i in range(len(subset)):
        prompt_middle += f"{i + 1}. {subset[i]}\n"

    output = gen_failure_instance(prompt_prefix + prompt_middle + prompt_suffix).replace('\n', '')
    print("Output:", output)

    with open("intersection_generated_failures.txt", "a") as f:
        f.write(f"{output}\n")
    
    iter += 1
    