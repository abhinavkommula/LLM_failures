import os
import statistics
from ast import literal_eval

threshold = 0.9
cur_dir = './data'
file_paths = []

for filename in os.listdir(cur_dir):
    file_path = os.path.join(cur_dir, filename)

    if os.path.isfile(file_path):
        words = filename.split('_')

        if len(words) == 2 and words[1] == 'failures.txt':
            file_paths.append(file_path)
     
for i in range(len(file_paths)):
    for j in range(len(file_paths)):
        if i == j:
            continue
        
        print(f"Transfer {file_paths[i]} -> {file_paths[j]}") 
        
        init_domain = {}
        transfer_domain = {}

        with open(file_paths[j], 'r') as f:
            for line in f:
                line = line.strip()
                tup_line = literal_eval(line)
                
                if len(tup_line[1]) == 0:
                    continue

                transfer_domain[tup_line[0]] = (tup_line[2], tup_line[3])

        is_failure = True

        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        failure_sim_scores = []
        failure_scores = []
        nonfailure_sim_scores = []
        nonfailure_scores = []

        with open(file_paths[i], 'r') as f:
            for line in f:
                line = line.strip()
                tup_line = literal_eval(line)

                if tup_line[2] >= threshold:
                    is_failure = False
        
                if len(tup_line[1]) == 0:
                    continue
            
                if tup_line[0] in transfer_domain:
                    if is_failure:
                        failure_sim_scores.append(transfer_domain[tup_line[0]][0])
                        failure_scores.append(transfer_domain[tup_line[0]][1])

                        if transfer_domain[tup_line[0]][0] < threshold:
                            true_positive += 1
                        else:
                            false_positive += 1
                    else:
                        nonfailure_sim_scores.append(transfer_domain[tup_line[0]][0])
                        nonfailure_scores.append(transfer_domain[tup_line[0]][1])

                        if transfer_domain[tup_line[0]][0] < threshold:
                            false_negative += 1
                        else:
                            true_negative += 1

        print(f"Failure Similarity Mean: {statistics.mean(failure_sim_scores)}")
        print(f"Nonfailure Similarity Mean: {statistics.mean(nonfailure_sim_scores)}")
        print(f"Failure Score Mean: {statistics.mean(failure_scores)}")
        print(f"Nonfailure Score Mean: {statistics.mean(nonfailure_scores)}")
       
        print(f"True Positive: {true_positive}")
        print(f"False Positive: {false_positive}")
        print(f"True Negative: {true_negative}")
        print(f"False Negative: {false_negative}")
        
        print(f"Failure Rate: {true_positive / (true_positive + false_positive)}")
        print(f"Nonfailure Rate: {false_negative / (true_negative + false_negative)}")
        print(f"Precision: {true_positive / (true_positive + false_positive)}")
        print(f"Recall: {true_positive / (true_positive + false_negative)}")

        print("===========================================")
