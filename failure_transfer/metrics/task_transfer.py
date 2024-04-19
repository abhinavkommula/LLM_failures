import csv
import os
import statistics
import itertools
import math

from ast import literal_eval

groups = [
    #[("translation", "squad_paragraphs_complex_french_paragraph"), ("translation", "squad_paragraphs_complex_spanish_paragraph"), ("translation", "squad_paragraphs_complex_dutch_paragraph")],

    #[("translation", "squad_paragraphs_disaster_french_paragraph"), ("translation", "squad_paragraphs_disaster_spanish_paragraph"), ("translation", "squad_paragraphs_disaster_dutch_paragraph")],

    [("translation", "squad_paragraphs_complex_spanish_paragraph"), ("summarization", "squad_paragraphs_complex"), ("completion", "squad_paragraphs_complex"), ("sentence_interweave", "squad_paragraphs_complex"), ("text_reorder", "squad_paragraphs_complex"), ("pronoun_replacement", "squad_paragraphs_complex")],

    #[("translation", "short_stories_french_paragraph"), ("translation", "short_stories_spanish_paragraph"), ("translation", "short_stories_dutch_paragraph")],
    #[("translation", "short_stories_spanish_paragraph"), ("summarization", "short_stories"), ("text_reorder", "short_stories"), ("sentence_interweave", "short_stories")],
    #[("style_gen", "short_stories_future"), ("style_gen", "short_stories_present"), ("style_gen", "short_stories_past")]
    #[("translation", "squad_paragraphs_french_paragraph"), ("translation", "squad_paragraphs_spanish_paragraph"), ("translation", "squad_paragraphs_dutch_paragraph")],
    #[("translation", "short-stories_bullet_french_paragraph"), ("translation", "short-stories_bullet_spanish_paragraph"), ("translation", "short-stories_bullet_dutch_paragraph")],
    #[("translation", "short-stories_bullet_spanish_paragraph"), ("summarization", "short-stories_bullet"), ("text_reorder", "short-stories_bullet"), ("sentence_interweave", "short-stories_bullet")],
]

def is_failure_translation(tup):
    if (len(tup[0]) / len(tup[1]) >= 2):
        return False
    
    return (tup[3] == 0) 
    
def is_failure_summarization(tup):
    try:
        res = (tup[2] == 0)
    except:
        print(tup)
        
    return (res)
    
def is_failure_style_gen(tup):
    return (tup[2] == 0)
    
def is_failure_text_reorder(tup):
    return (tup[2] >= 0.35)

def is_failure_sentence_interweave(tup):
    return ((tup[4] + tup[5]) / 2 <= 0.6)

def is_failure_gpt_eval_binary(tup):
    return (tup[2] == 0)

def is_failure_gpt_eval_score(tup):
    return (tup[2] < 7)

def is_failure_pronoun_replacement(tup):
    return (tup[2] >= 0.35)

def all_subsets(lst):
    subsets = []
    for i in range(len(lst) + 1):
        subsets.extend(list(itertools.combinations(lst, i)))
    return subsets

def form(f):
    return (float('{:,.3f}'.format(f)))


failure_function_mapping = {
    "translation": is_failure_translation,
    "summarization": is_failure_summarization,
    "style_gen": is_failure_style_gen,
    "text_reorder": is_failure_text_reorder,
    "sentence_interweave": is_failure_sentence_interweave, 
    "ambiguity": is_failure_gpt_eval_binary, 
    "argument": is_failure_gpt_eval_score, 
    "passive": is_failure_gpt_eval_binary, 
    "sentence_succinct": is_failure_gpt_eval_score, 
    "completion": is_failure_gpt_eval_binary, 
    "pronoun_replacement": is_failure_pronoun_replacement, 
}

suffix_mapping = {
    "translation": "_failures.txt",
    "summarization": "_failures.txt",
    "style_gen": "_failures.txt",
    "text_reorder": "_failures_similarity.txt",
    "sentence_interweave": "_failures_similarity.txt", 
    "ambiguity": "_failures.txt", 
    "argument": "_failures.txt", 
    "passive": "_failures.txt", 
    "sentence_succinct": "_failures.txt", 
    "completion": "_failures.txt", 
    "pronoun_replacement": "_failures.txt"
}

results = []
cnv = {}

for group in groups:    
    for output_file_tup in group:  
        ar = []
        
        for file_tup in group:
            if file_tup == output_file_tup:
                continue

            ar.append(file_tup)
        
        output_failures = 0
        total_output = 0

        output_path = output_file_tup[0] + "/data/" + output_file_tup[1] + suffix_mapping[output_file_tup[0]]
        output_is_failure = failure_function_mapping[output_file_tup[0]]
        instance_to_tup = {}
        
        with open(output_path, 'r') as f:
            for line in f:
                line = line.strip()
                tup_line = literal_eval(line)
                    
                if len(tup_line[1]) == 0:
                    continue
                
                instance_to_tup[tup_line[0]] = tup_line
                
                if output_is_failure(tup_line):                 
                    output_failures += 1
                
                total_output += 1
        
        for subset in all_subsets(ar):
            if len(subset) > 1 or len(subset) == 0:
                continue
            
            print(f"Transfer {subset} -> {output_file_tup}")
            
            input_instance_failures = {}
            input_instance_nonfailures = {}
            subset_str = ""

            for file_tup in subset:
                subset_str += (file_tup[0] + '_' + file_tup[1] + ';')
                
                path = file_tup[0] + "/data/" + file_tup[1] + suffix_mapping[file_tup[0]]                                
                failure_handler = failure_function_mapping[file_tup[0]]

                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        tup_line = literal_eval(line)
                            
                        if len(tup_line[1]) == 0:
                            continue
                            
                        if failure_handler(tup_line):   
                            if tup_line[0] not in input_instance_failures:
                                input_instance_failures[tup_line[0]] = 0
                            
                            input_instance_failures[tup_line[0]] += 1
                        else:
                            if tup_line[0] not in input_instance_nonfailures:
                                input_instance_nonfailures[tup_line[0]] = 0
                            
                            input_instance_nonfailures[tup_line[0]] += 1

            true_positive = 0
            false_positive = 0
            true_negative = 0
            false_negative = 0  

            failures = []
            nonfailures = []
            transfers = []
            
            for instance in input_instance_failures:
                if abs(input_instance_failures[instance] - len(subset)) <= 0:
                    failures.append(instance)
            
            for instance in input_instance_nonfailures:
                if abs(input_instance_nonfailures[instance] - len(subset)) <= 0:
                    nonfailures.append(instance)
            
            for fail in failures:
                if fail not in instance_to_tup:
                    continue
                
                if output_is_failure(instance_to_tup[fail]):
                    true_positive += 1
                    transfers.append(fail)
                else:
                    false_positive += 1
                
            for nonfail in nonfailures:
                if nonfail not in instance_to_tup:
                    continue
                
                if output_is_failure(instance_to_tup[nonfail]):
                    false_negative += 1
                else:
                    true_negative += 1
                    
            transfers.sort(key = lambda text: len(text))

            #with open(f"./transfer/{subset_str}-{output_file_tup[0]}_{output_file_tup[1]}_transfer.txt", "w") as f:
            #    for transfer in transfers:
            #        f.write(f"{transfer}\n")
            
            #with open(f"transfer/{subset_str}-{output_file_tup[0]}_{output_file_tup[1]}_results.csv", "w") as f:
            #    f.write("NOTHING...\n")

            print(f"True Positive: {true_positive}")
            print(f"False Positive: {false_positive}")
            print(f"True Negative: {true_negative}")
            print(f"False Negative: {false_negative}\n")
            
            print(f"Output Failure Rate: {form(output_failures / total_output)}")
            print(f"Transfer Failure Rate: {form(true_positive / max(1, (true_positive + false_positive)))}")
            print(f"Transfer Nonfailure Rate: {form(false_negative / max(1, (true_negative + false_negative)))}")
            print(f"Precision: {form(true_positive / max(1, (true_positive + false_positive)))}")
            print(f"Recall: {form(true_positive / max(1, (true_positive + false_negative)))}")
            print("========================================")

            ratio = (true_positive / max(1, (true_positive + false_positive)))

            transfer_str = f"""True Positive: {true_positive}
False Positive: {false_positive}
True Negative: {true_negative}
False Negative: {false_negative}

Output Failure Rate: {form(output_failures / total_output)}
Transfer Failure Rate: {form(true_positive / max(1, (true_positive + false_positive)))}
Confidence Interval: {form(ratio - 1.96/math.sqrt(len(failures) + len(nonfailures)))} - {form(ratio + 1.96/math.sqrt(len(failures) + len(nonfailures)))}
Transfer Nonfailure Rate: {form(false_negative / max(1, (true_negative + false_negative)))}
Precision: {form(true_positive / max(1, (true_positive + false_positive)))}
Recall: {form(true_positive / max(1, (true_positive + false_negative)))}
            """

            if subset[0] not in cnv:
                cnv[subset[0]] = {}
                
            cnv[subset[0]][output_file_tup] = transfer_str

            results.append((output_failures / total_output, true_positive / max(1, (true_positive + false_positive)), subset, output_file_tup))

print("Top 10 sorted by ratio:")
results.sort(key = lambda t : -(t[1] / t[0]))

for i in range(10):
    print(f"{i + 1}. {results[i]}")

print("Top 10 sorted by difference:")
results.sort(key = lambda t : -(t[1] - t[0]))

for i in range(10):
    print(f"{i + 1}. {results[i]}")

for iter in range(len(groups)):
    group = groups[iter]
    data = []
    data.append(["N/A"])
    
    for i in range(len(group)):
        data[0].append(group[i][0])

    for i in range(len(group)):
        data.append([])
        data[-1].append(group[i][0])
        
        for j in range(len(group)):
            if i == j:
                data[-1].append("N/A")
            else:
                data[-1].append(cnv[group[i]][group[j]])
    
    print("Data:", data)

    with open(f'transfer/task_transfer_output_{iter}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)