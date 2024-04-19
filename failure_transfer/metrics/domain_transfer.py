import csv
import os
import statistics
import itertools
from ast import literal_eval

original = []

with open('/accounts/projects/jsteinhardt/akommula/LLM_failures/failure_transfer/metrics/pronoun_replacement/data/squad_paragraphs_complex_failures.txt', 'r') as f:
    for line in f:
        tup = literal_eval(line)
        original.append(tup)

groups = [
    [("translation", "squad_paragraphs_complex_spanish_paragraph"), ("translation", "squad_paragraphs_disaster_spanish_paragraph")],
    [("summarization", "squad_paragraphs_complex"), ("summarization", "squad_paragraphs_disaster")], 
    [("completion", "squad_paragraphs_complex"), ("completion", "squad_paragraphs_disaster")],
    [("sentence_interweave", "squad_paragraphs_complex"), ("sentence_interweave", "squad_paragraphs_disaster")],
    [("text_reorder", "squad_paragraphs_complex"), (("text_reorder", "squad_paragraphs_disaster"))],
    [("pronoun_replacement", "squad_paragraphs_complex"), ("pronoun_replacement", "squad_paragraphs_disaster")],
        
    #[("translation", "short_stories_spanish_paragraph"), ("translation", "short-stories_bullet_spanish_paragraph"), ("translation", "short-stories_dialogue_spanish_paragraph")],
    #[("summarization", "short_stories"), ("summarization", "short-stories_bullet"), ("summarization", "short-stories_dialogue")],
    #[("text_reorder", "short_stories"), ("text_reorder", "short-stories_bullet"), ("text_reorder", "short-stories_dialogue")],
    #[("sentence_interweave", "short_stories"), ("sentence_interweave", "short-stories_bullet"), ("sentence_interweave", "short-stories_dialogue")],    
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

mapping = {}
def find_domain_mappings():    
    mapping["squad"] = {}
    
    for i in range(len(original)):
        mapping["squad"][original[i]] = original[i]
    
    mapping["squad_disaster"] = {}

    with open("/accounts/projects/jsteinhardt/akommula/LLM_failures/failure_transfer/data/domain_shift/squad_paragraphs_disaster_examples.txt", "r") as f:
        lines = f.readlines()
        idx = 0

        for i in range(0, len(lines), 4):
            group = lines[i:i+4]
            grouped_string = ''.join(group)
            
            if idx < len(original):
                mapping["squad_disaster"][grouped_string] = original[idx]
                idx += 1            

print("Calculating domain mappings...")
find_domain_mappings()

def fn_mapping(file_name):
    if "squad_paragraphs_complex":
        return mapping["squad"]
    elif "squad_paragraphs_disaster":
        return mapping["squad_disaster"]
    
    return None

'''
    if "short-stories_bullet" in file_name:
        return mapping["short-stories_bullet"]
    elif "short-stories_dialogue" in file_name:
        return mapping["short-stories_dialogue"]
    elif "short_stories" in file_name:
        return mapping["short_stories"]
'''    
    
failure_function_mapping = {
    "translation": is_failure_translation,
    "summarization": is_failure_summarization,
    "style_gen": is_failure_style_gen,
    "text_reorder": is_failure_text_reorder,
    "sentence_interweave": is_failure_sentence_interweave,
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
        
        find_original = fn_mapping(output_file_tup[1])
        
        with open(output_path, 'r') as f:
            for line in f:
                line = line.strip()
                tup_line = literal_eval(line)
                    
                if len(tup_line[1]) == 0:
                    continue
            
                if tup_line[0] in find_original:
                    instance_to_tup[find_original[tup_line[0]]] = tup_line
                    
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
                find_original = fn_mapping(file_tup[1])

                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        tup_line = literal_eval(line)
                            
                        if len(tup_line[1]) == 0:
                            continue
                        
                        if tup_line[0] in find_original:
                            orig_tup = find_original[tup_line[0]]

                            if failure_handler(tup_line):                               
                                if orig_tup not in input_instance_failures:
                                    input_instance_failures[orig_tup] = 0
                                
                                input_instance_failures[orig_tup] += 1
                            else:
                                if orig_tup not in input_instance_nonfailures:
                                    input_instance_nonfailures[orig_tup] = 0
                                
                                input_instance_nonfailures[orig_tup] += 1

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
            
            #with open(f"stransfer/{subset_str}-{output_file_tup[0]}_{output_file_tup[1]}_results.csv", "w") as f:
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

            results.append((output_failures / total_output, true_positive / max(1, (true_positive + false_positive)), subset, output_file_tup))

print("Top 10 sorted by ratio:")
results.sort(key = lambda t : -(t[1] / t[0]))

for i in range(10):
    print(f"{i + 1}. {results[i]}")

print("Top 10 sorted by difference:")
results.sort(key = lambda t : (-t[1] - t[0]))

for i in range(10):
    print(f"{i + 1}. {results[i]}")