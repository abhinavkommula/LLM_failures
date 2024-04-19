import os
from ast import literal_eval

cur_dir = './data/'

def round3(x):
    return float('{:,.3f}'.format(x))

paths = []
for filename in os.listdir(cur_dir):
    file_path = os.path.join(cur_dir, filename)

    if os.path.isfile(file_path):
        words = filename.split('_')

        if words[-1] == 'failures.txt' and "adaptive" not in words:               
            paths.append(file_path)

paths.sort()

for file_path in paths:            
    sim_scores = []
    gpt_scores = []

    print("Examining file_path:", file_path)
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            tup_line = literal_eval(line)

            if tup_line[1] == '':
                continue

            sim_scores.append(tup_line[2])
            gpt_scores.append(tup_line[3])

    sim_scores.sort()
    gpt_scores.sort()

    idx_sim = 0
    while(idx_sim < len(sim_scores) and sim_scores[idx_sim] <= 0.9):
        idx_sim += 1

    idx_gpt = 0
    while(idx_gpt < len(gpt_scores) and gpt_scores[idx_gpt] == 0):
        idx_gpt += 1

    if idx_sim == 0 and idx_gpt == 0:
        continue

    print(f"Mean similarity score: {round3(sum(sim_scores) / max(1, len(sim_scores)))}")
    print(f"Mean gpt score: {round3(sum(gpt_scores) / max(1, len(gpt_scores)))}")
    print(f"Failure Rate Similarity: {round3(idx_sim / max(1, len(sim_scores)))}")
    print(f"Failure Rate GPT: {round3(idx_gpt / max(1, len(gpt_scores)))}")

    print("====================================")
