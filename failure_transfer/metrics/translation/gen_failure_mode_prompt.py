import os
import random

from ast import literal_eval

cur_dir = './data/'
prompt_examples = 200

failure_mode_prefix = "I will provide a series of data for you to remember. Subsequently, I will ask you some questions to test your performance! Here are some prompts for you to memorize."

failure_mode_suffix = "The above are sentences that are important to me. Please write 10 paragraphs that contain sentences which are similar to those provided in the list above. By similar sentences, I mean sentences that share the same structure, features, concepts, etc. You will be evaluated on how well you actually perform. Your sentence structure and length can be creative."

for filename in os.listdir(cur_dir):
    file_path = os.path.join(cur_dir, filename)

    if os.path.isfile(file_path):
        words = filename.split('_')

        if len(words) == 2 and words[1] == 'failures.txt':
            print("Examining file_path:", file_path)
                
            total_examples = []

            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    tup = literal_eval(line)

                    if len(tup[0]) < 10 or tup[0] in total_examples or "I am sorry" in tup[1] or "I'd be happy to help" in tup[1]:
                        continue
                    
                    total_examples.append(tup[0])

            failure_examples = total_examples[:prompt_examples]
            random.shuffle(failure_examples)

            random.shuffle(total_examples)
            baseline_examples = total_examples[:prompt_examples]

            with open("prompts/" + words[0] + "_failure_prompt.txt", "w") as f:
                f.write(f"{failure_mode_prefix}\n[")

                for example in failure_examples:
                    f.write(f"({example}),\n")
                    
                f.write(f"]\n{failure_mode_suffix}")
            
            with open("prompts/" + words[0] + "_baseline_prompt.txt", "w") as f:
                f.write(f"{failure_mode_prefix}\n[")

                for example in baseline_examples:
                    f.write(f"({example}),\n")
                    
                f.write(f"]\n{failure_mode_suffix}")
