import os
from ast import literal_eval

cur_dir = './data/'

for filename in os.listdir(cur_dir):
    file_path = os.path.join(cur_dir, filename)

    if os.path.isfile(file_path):
        words = filename.split('_')

        if len(words) == 2 and words[1] == 'failures.txt':
            print("Examining file_path:", file_path)
                
            sim_scores = []
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    tup_line = literal_eval(line)

                    sim_scores.append(tup_line[2])
            
            print(words[0], ":", sum(sim_scores) / max(1, len(sim_scores)))
