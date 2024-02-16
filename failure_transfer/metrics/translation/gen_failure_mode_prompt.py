import os
import random

from ast import literal_eval

cur_dir = './data/'
prompt_examples = 200

failure_mode_prefix = "I will provide a series of data for you to remember. Subsequently, I will ask you some questions to test your performance! Here are some prompts for you to memorize."

failure_mode_suffix = "I am trying to find failures with a language model. The above are sentences that a language model fails to translate accurately. Using these specific examples, are there any general type of failures you notice the language model is making, or any common features of sentences that the language model fails to translate? Try to give failures that are specific enough that someone could reliably produce examples that the language model would fail to translate. Please try to give as many general failures as possible. Please focus on what you notice on the text, rather than what you know about languages and translation tasks. In your failure modes, please explain clearly why the failure would lead to problems for future tasks related to translation. Please summarize as many as you can and stick to the examples rather than any prior knowledge."


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
