from translation_pipeline import Translation
from information_retrieval_pipeline import InformationRetrieval
from summarization_score_pipeline import SummarizationScore
from reasoning_pipeline import Reasoning
from interact_llama import InteractLLaMA

failure_modes = []

with open("failure_modes.txt", 'r') as f:
    for line in f:
        failure_modes.append(line.strip().replace('\n', ''))

tasks = [ 
    (Reasoning, {}),
    (Translation, {"language": "Chinese", "threshold": 0.9}),
    (SummarizationScore, {}), 
    #(InformationRetrieval, {"num_facts": 3}),
]

interacter = InteractLLaMA()
num_examples = 100
all_metrics = []

for failure_mode in failure_modes:
    metrics = []

    for task_type, params in tasks:
        instance = task_type(failure_mode, num_examples, interacter, **params)
        instance.gen_data()
        instance.pipeline()

        metrics.append(instance.extract_metrics())
   
    print(metrics)
    all_metrics.append(metrics)

with open("metrics/aggregate.txt", 'w') as f:
    for i in range(len(failure_modes)):
        f.write(f"Failure mode: {failure_modes[i]}\nRates: {all_metrics[i]}\n")

