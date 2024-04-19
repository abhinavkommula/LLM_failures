from ast import literal_eval
import itertools

# Idea 1: Generalist Failure Instances
# Take all intersection of failures, use random subsets of 3 to generate subsequent examples

tasks = [("translation", "spanish_paragraph"), ("summarization", "short_stories"), ("style_gen", "poetic"), ("text_reorder", "short_stories"), ("sentence_interleave", "short_stories")]

def is_failure_translation(tup):
    return (tup[3] == 0) 
    
def is_failure_summarization(tup):
    return (tup[10] <= 0.3)
    
def is_failure_style_gen(tup):
    return (tup[11] == 0)
    
def is_failure_text_reorder(tup):
    return (tup[2] >= 0.35)

def is_failure_sentence_interleave(tup):
    return (max(tup[-4:]) <= 0.6)

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
    "sentence_interleave": is_failure_sentence_interleave
}

suffix_mapping = {
    "translation": "_failures.txt",
    "summarization": "_failures_precision.txt",
    "style_gen": "_failures_precision.txt",
    "text_reorder": "_failures_similarity.txt",
    "sentence_interleave": "_failures_similarity.txt"
}

intersections = {}

for task in tasks:
    read_path = task[0] + "/data/" + task[1] + suffix_mapping[task[0]]
    is_failure = failure_function_mapping[task[0]]
    
    with open(read_path, "r") as f:
        for line in f:
            line = line.strip()
            tup_line = literal_eval(line)
            
            if len(tup_line[1]) == 0:
                continue
            
            if is_failure(tup_line):
                if tup_line[0] not in intersections:
                    intersections[tup_line[0]] = []
                
                intersections[tup_line[0]].append(task[0])

intersections = sorted(intersections.items(), key = lambda t : -len(t[1]))

with open("aggregates/intersections.txt", "w") as f:
    for key, value in intersections:
        f.write(f"({key}\n{value})\n")
            

# Idea 2: Clustering weak vs strong examples
# Take random  


