import os
import statistics
from ast import literal_eval

threshold = 1
cur_dir = './data'
file_paths = []

languages = ["french", "spanish", "dutch"] 

for filename in os.listdir(cur_dir):
    file_path = os.path.join(cur_dir, filename)

    if os.path.isfile(file_path):
        words = filename.split('_')

        if len(words) == 3 and words[1] == "adaptive" and words[-1] == 'failures.txt' and words[0].lower() in languages:
            file_paths.append(file_path)

prompt_strength = {}

for i in range(len(file_paths)):
    with open(file_paths[i], 'r') as f:
        for line in f:
            line = line.strip()
            tup_line = literal_eval(line)

            if len(tup_line[1]) == 0:
                continue
            
            if tup_line[0] not in prompt_strength:
                prompt_strength[tup_line[0]] = []

            prompt_strength[tup_line[0]].append(float(tup_line[2]))

    for j in range(len(file_paths)):
        if i == j:
            continue

        domain_1 = file_paths[i][len(cur_dir) + 1:].split('_')[0]
        domain_2 = file_paths[j][len(cur_dir) + 1:].split('_')[0]
        
        print(f"Transfer {domain_1} -> {domain_2}") 
        
        init_domain = {}
        transfer_domain = {}

        with open(file_paths[j], 'r') as f:
            for line in f:
                line = line.strip()
                tup_line = literal_eval(line)
                
                if len(tup_line[1]) == 0:
                    continue

                transfer_domain[tup_line[0]] = (tup_line[2], tup_line[3])

        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        failure_sim_scores = []
        failure_scores = []
        nonfailure_sim_scores = []
        nonfailure_scores = []
        
        transfers = []

        with open(file_paths[i], 'r') as f:
            for line in f:
                line = line.strip()
                tup_line = literal_eval(line)

                if len(tup_line[1]) == 0:
                    continue
            
                if tup_line[0] in transfer_domain:
                    if tup_line[3] < threshold:
                        failure_sim_scores.append(transfer_domain[tup_line[0]][0])
                        failure_scores.append(transfer_domain[tup_line[0]][1])

                        if transfer_domain[tup_line[0]][1] < threshold:
                            true_positive += 1
                            transfers.append((tup_line[0], tup_line[2], transfer_domain[tup_line[0]][0], 1))
                        else:
                            false_positive += 1
                    else:
                        nonfailure_sim_scores.append(transfer_domain[tup_line[0]][0])
                        nonfailure_scores.append(transfer_domain[tup_line[0]][1])

                        if transfer_domain[tup_line[0]][1] < threshold:
                            false_negative += 1
                            transfers.append((tup_line[0], tup_line[2], transfer_domain[tup_line[0]][0], 0))
                        else:
                            true_negative += 1

        transfers.sort(key = lambda t : max(t[1], t[2]))
        with open(f"./data/{domain_1}_{domain_2}_transfer.txt", "w") as f:
            for transfer in transfers:
                f.write(f"{transfer}\n")

        def form(f):
            return (float('{:,.3f}'.format(f)))

        print(f"Failure: {form(statistics.mean(failure_sim_scores))}")
        print(f"Nonfailure: {form(statistics.mean(nonfailure_sim_scores))}")
        print(f"Failure: {form(statistics.mean(failure_scores))}")
        print(f"Nonfailure: {form(statistics.mean(nonfailure_scores))}")
       
        print(f"True Positive: {true_positive}")
        print(f"False Positive: {false_positive}")
        print(f"True Negative: {true_negative}")
        print(f"False Negative: {false_negative}")
        
        print(f"Failure Rate: {form(true_positive / (true_positive + false_positive))}")
        print(f"Nonfailure Rate: {form(false_negative / (true_negative + false_negative))}")
        print(f"Precision: {form(true_positive / (true_positive + false_positive))}")
        print(f"Recall: {form(true_positive / (true_positive + false_negative))}")

        print("===========================================")


prompt_strength = {k: v for k, v in sorted(prompt_strength.items(), key = lambda item : max(item[1]))}

idx = 0
num_examples = 75 

with open(f"./prompts/gpt_prompt_transfer.txt", "w") as f:
    f.write("I will provide a series of data for you to remember. Subsequently, I will ask you some questions to test your performance! Here are some sentences for you to memorize.\n")
    f.write("Weak translations: [")

    for prompt in prompt_strength:
        if len(prompt_strength[prompt]) != 3:
            continue

        #f.write(f"({prompt}, {prompt_strength[prompt]}),\n")
        f.write(f"({prompt}),\n")

        idx += 1
        if idx == num_examples:
            break
    
    f.write(f"]\n")
    f.write("Strong translations: [")

    prompt_strength = {k: v for k, v in sorted(prompt_strength.items(), key = lambda item : -max(item[1]))}
    idx = 0

    for prompt in prompt_strength:
        if len(prompt_strength[prompt]) != 3 or len(prompt) < 120:
            continue

        #f.write(f"({prompt}, {prompt_strength[prompt]}),\n")
        f.write(f"({prompt}),\n")

        idx += 1
        if idx == num_examples:
            break

    f.write(f"]\n")
    f.write("I am trying to find failures with a language model's ability to translate. I have provided sentences that it consistently scores poorly on, along with sentences that it consistently scores strongly on. Using these specific examples, are there any general types of failures you notice the language model is making, or any common features that the language model is unable to translate well? Focus on failures or common features that are present amongst the weak translations, but not present amongst the strong translations. Try to give failures that are specific enough that someone could reliably produce examples that the language model would translate poorly. Please summarize as many as you can and stick to the examples. Remember, the general features you provide should be present in the weak examples but not present in the strong examples.\n")
    
