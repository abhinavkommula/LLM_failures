from translation_pipeline import Translation
from information_retrieval_pipeline import InformationRetrieval
from summarization_score_pipeline import SummarizationScore

failure_modes = []

with open("failure_modes.txt", 'r') as f:
    for line in f:
        failure_modes.append(line.strip().replace('\n', ''))

tasks = [ 
    (SummarizationScore, {})
    (InformationRetrieval, {"num_facts": 5}),
    (Translation, {"language": "French", "threshold": 0.9})
]

interacter = InteractLLaMA()
num_examples = 10
all_metrics = []

for failure_mode in failure_modes:
    metrics = []

    for task_type, params in tasks:
        instance = task_type(failure_mode, num_examples, **params)
        instance.gen_data()
        instance.pipeline()

        metrics.append(instance.extract_metrics())
    
    all_metrics.append(metrics)

