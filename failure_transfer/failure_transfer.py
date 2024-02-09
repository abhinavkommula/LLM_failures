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
    (Translation, {"name": "french", "language": "French", "threshold": 2.0}),
    (Translation, {"name": "french_spanish_transfer", "language": "Spanish", "threshold": 2.0, "read_file": "metrics/translation/french_failures.txt"}),
    (Translation, {"name": "spanish", "language": "Spanish", "threshold": 2.0}),
    (Translation, {"name": "spanish_french_transfer", "language": "French", "threshold": 2.0, "read_file": "metrics/translation/spanish_failures.txt"}),
    (Translation, {"name": "spanish_chinese_transfer", "language": "Chinese", "threshold": 2.0, "read_file": "metrics/translation/spanish_failures.txt"}),
    #(SummarizationScore, {"name": "}),
]

interacter = InteractLLaMA()
num_examples = 150
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

