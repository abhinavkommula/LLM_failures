from translation_pipeline import Translation
from information_retrieval_pipeline import InformationRetrieval
from summarization_score_pipeline import SummarizationScore
from interact_llama import InteractLLaMA

failure_modes = []

with open("failure_modes.txt", 'r') as f:
    for line in f:
        failure_modes.append(line.strip().replace('\n', ''))

tasks = [ 
    (SummarizationScore, {}), 
    (Translation, {"language": "French", "threshold": 0.9})
    #(InformationRetrieval, {"num_facts": 5}),
]

interacter = InteractLLaMA()
num_examples = 250
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

