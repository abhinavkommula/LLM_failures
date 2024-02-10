from translation_pipeline import Translation
from information_retrieval_pipeline import InformationRetrieval
from summarization_score_pipeline import SummarizationScore
from reasoning_pipeline import Reasoning
from interact_llama import InteractLLaMA

failure_modes = []

with open("failure_modes.txt", 'r') as f:
    for line in f:
        failure_modes.append(line.strip().replace('\n', ''))

short_stories_samples = [
    "One day there was a clever radio. It sang lovely songs and made everyone happy. But then something strange happened. The radio began to split in two. It split and split until the pieces were very small. Everyone was so sad to see their clever radio fall apart. But then a very special thing happened. A man came and fixed the radio, and soon it was singing again. Everyone was so happy and cheered that their clever radio was back!", 
    "Bob and Sue were going to a party. They were celebrating Sue's birthday. When they arrived, Bob started kicking the ball around. He was having lots of fun. But before long, Sue came over and said, \"No, Bob, don\'t be so foolish. It\'s my birthday party!\" Bob stopped to listen, then gave the ball a big kick. He and Sue laughed and ran around the party together. Everyone had a great time.",
]

news_articles_samples = [
    "Tomas Medina Caracas was a fugitive from a U.S. drug trafficking indictment . \"El Negro Acacio\" allegedly helped manage extensive cocaine network . U.S. Justice Department indicted him in 2002 . Colombian military: He was killed in an attack on a guerrilla encampment .",
    "No bail for ex-NFL star accused of directing men in alleged armed robbery . Simpson faces charges of robbery, assault, burglary and conspiracy . Alleged robbery involved sports-related items, police say . Simpson arrested Sunday in Las Vegas, but he says items were his .",
    "India elects first female president, official results show Saturday . Pratibha Patil's supporters are calling victory a boost for women's rights . Bitter election campaign was marked by scandal . 72-year-old Patil was the ruling coalition's nominee for mainly ceremonial post ."
]

tasks = [ 
    #(Translation, {"name": "french", "language": "French", "threshold": 2.0}),
    #(Translation, {"name": "french_spanish_transfer", "language": "Spanish", "threshold": 2.0, "read_file": "metrics/translation/french_failures.txt"}),
    #(Translation, {"name": "spanish", "language": "Spanish", "threshold": 2.0}),
    #(Translation, {"name": "spanish_french_transfer", "language": "French", "threshold": 2.0, "read_file": "metrics/translation/spanish_failures.txt"}),
    #(Translation, {"name": "chinese", "language": "Mandarin Chinese", "threshold": 2.0}),
    #(Translation, {"name": "spanish_chinese_transfer", "language": "Mandarin Chinese", "threshold": 2.0, "read_file": "metrics/translation/spanish_failures.txt"}),
    #(Translation, {"name": "arabic", "language": "Arabic", "threshold": 2.0}),
    #(Translation, {"name": "spanish_arabic_transfer", "language": "Arabic", "threshold": 2.0, "read_file": "metrics/translation/spanish_failures.txt"})

    (SummarizationScore, {"name": "news", "domain": "news articles"}),
    #(SummarizationScore, {"name": "news_short_stories_transfer", "domain": "short stories", 
    #                      "example1": short_stories_samples[0], "example2": short_stories_samples[1], "read_file": "metrics/summarization/news_failures.txt"}),

    (SummarizationScore, {"name": "short_stories", "domain": "short stories"}),
    #(SummarizationScore, {"name": "short_stories_news_transfer", "domain": "news articles", "read_file": "metrics/summarization/short_stories_failures.txt"}),
]

interacter = InteractLLaMA()
num_examples = 200
all_metrics = []

for failure_mode in failure_modes:
    metrics = []

    for task_type, params in tasks:
        print(f"Starting task: {task_type} with params: {params}...")

        instance = task_type(failure_mode, num_examples, interacter, **params)
        instance.gen_data()
        instance.pipeline()

        metrics.append(instance.extract_metrics())
   
    print(metrics)
    all_metrics.append(metrics)

with open("metrics/aggregate.txt", 'w') as f:
    for i in range(len(failure_modes)):
        f.write(f"Failure mode: {failure_modes[i]}\nRates: {all_metrics[i]}\n")

