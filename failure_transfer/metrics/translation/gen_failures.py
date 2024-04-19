import openai
import re

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
    except:
        data = "Error Querying OpenAI"
        exit()

    return (data)

def gen_failures(context, failure_mode, gen_examples, ref, num_paragraphs = 5, temp = 0.7, model = 'gpt-4-turbo'):
    if type(failure_mode) == list:
        assert(len(failure_mode) == 2)
        query = context + "\nFailure Mode #1:[\n" + failure_mode[0] + "\n]\nFailure Mode #2: [\n" + failure_mode[1] + "\n]"
    else:
        query = context + "\nFailure Mode: [\n" + failure_mode + "\n]"

    messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': query}]
    failures = []
    
    for i in range(int(gen_examples // num_paragraphs)):
        print(f"QUERYING BATCH {i}...")
            
        llm_output = run_gpt(messages, model, max_tokens = 1000, temperature = temp)
        paragraphs = re.split(r'\d+\)|\d+\.|Paragraph \d+', llm_output)[1:]
        
        if len(paragraphs) != num_paragraphs:
            print("Error with parsing following output:")
            print("LLM Output:", llm_output)
            print("Paragraphs:", paragraphs)
            print(len(paragraphs), num_paragraphs)

            continue
        
        for par in paragraphs:
            if par not in ref:
                failures.append(par)
                            
        print(paragraphs)

    return (failures)

failures = []
failure_modes = []
examples_per_failure_mode = 200
paired_failure_modes = True

input_path = "./failure_modes_paired.txt"
output_path = "./data/translation_failures_paired_"

if paired_failure_modes:
    prompt = "Write down 5 sentences that a language model might struggle to translate accurately due to the following failure modes. Do not include explanations, just the sentences. "
else:
    #prompt = "I will provide you with an explanation of the key differences between weak and strong sentences. Using these examples and the description of the differences, generate 5 new weak sentences. Be creative and vary the length of the sentences, but only write grammatically correct sentences and do not include an explanation. "
    prompt = "Write down 5 additional sentences that a language model with the following failure mode would have difficulty translating accurately. Only write the 5 sentences, do not include an explanation. "

with open(input_path, "r") as f:
    for line in f:
        failure_mode = line.strip().replace('\n', '')
        failure_modes.append(failure_mode)          

    if paired_failure_modes:
        for i in range(len(failure_modes)):
            for j in range(i + 1, len(failure_modes)):
                print(f"Generating Failure Modes: {failure_modes[i]}, {failure_modes[j]}")
                
                remaining = examples_per_failure_mode
                
                failures.append([])
                while (remaining > 0):
                    generated = gen_failures(prompt, [failure_modes[i], failure_modes[j]], remaining + 5, failures)     
                    failures[-1].extend(generated)
                    
                    remaining -= len(generated)
                
                failures[-1] = failures[-1][:examples_per_failure_mode]
                
    else:
        for failure_mode in failure_modes:
            print(f"\nGenerating failure mode: {failure_mode}\n")
            
            remaining = examples_per_failure_mode
        
            failures.append([])
            while (remaining > 0):
                generated = gen_failures(prompt, failure_mode, remaining + 5, failures)
                failures[-1].extend(generated)
                
                remaining -= len(generated)
            
            failures[-1] = failures[-1][:examples_per_failure_mode]

if paired_failure_modes:
    idx = 0
    
    for i in range(len(failure_modes)):
        for j in range(i + 1, len(failure_modes)):
            with open(output_path + str(i) + str(j) + ".txt", "w") as f:
                print(f"Writing to: {output_path + str(i) + str(j) + '.txt'}")
                
                for failure in failures[idx]:
                    failure = failure.replace('\n', '').strip()
                    f.write(f"{failure}\n")
            
                idx += 1
                 
else:
    for i in range(len(failure_modes)):
        with open(output_path + str(i) + ".txt", "w") as f:  
            print(f"Writing to: {output_path + str(i) + '.txt'}")

            for failure in failures[i]:
                failure = failure.replace('\n', '').strip()
                f.write(f"{failure}\n")