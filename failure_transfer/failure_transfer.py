from translation_pipeline import Translation
from information_retrieval_pipeline import InformationRetrieval
from summarization_score_pipeline import SummarizationScore
from reasoning_pipeline import Reasoning
from logger import Logger
from interact_llama import InteractLLaMA
from sentence_generation import SquadSentenceDataset

from scrape.scrape_news import NewsScrape
from scrape.scrape_stories import StoriesScrape
from scrape.scrape_arxiv import ArxivScrape
from scrape.scrape_pubmed import PubmedScrape

import random

failure_modes = []

with open("failure_modes.txt", 'r') as f:
    for line in f:
        failure_modes.append(line.strip().replace('\n', ''))

translation = False
summarization = True

assert((translation or summarization) == True)

logger = Logger()
main_logger = Logger()
main_logger.change_job("main")

interacter = InteractLLaMA(logger)

sentences = []
tasks = []

if summarization:
    total_examples = 10000

    news_scraper = NewsScrape()
    stories_scraper = StoriesScrape()
    #arxiv_scraper = ArxivScrape()
    #pubmed_scraper = PubmedScrape()

if translation:
    total_examples = 50000
    subset_size = 10000
    
    ssd = SquadSentenceDataset('train')
    ssd_gen = ssd.sentence_generator()

    for i in range(total_examples):
        sentences.append(next(ssd_gen))
    
    random.shuffle(sentences)
    sentences_subset = sentences[:subset_size]
    
    tasks.extend([
        (Translation, {"initial_domain": sentences_subset, "name": "french", "language": "French", "threshold": 2.0}),
        (Translation, {"initial_domain": sentences_subset, "name": "spanish", "language": "Spanish", "threshold": 2.0}),
        (Translation, {"initial_domain": sentences_subset, "name": "dutch", "language": "Dutch", "threshold": 2.0}),
        
        #(Translation, {"name": "french_spanish_transfer", "language": "Spanish", "threshold": 0.9, "read_file": "metrics/translation/data/french_failures.txt"}), 
        #(Translation, {"name": "french_dutch_transfer", "language": "Dutch", "threshold": 0.9, "read_file": "metrics/translation/data/french_failures.txt"}),
        #(Translation, {"name": "french_arabic_transfer", "language": "Arabic", "threshold": 2.0, "read_file": "metrics/translation/data/french_failures.txt"}),
        #(Translation, {"name": "french_chinese_transfer", "language": "Mandarin Chinese", "threshold": 2.0, "read_file": "metrics/translation/data/french_failures.txt"}),
        #(Translation, {"name": "french_korean_transfer", "language": "Korean", "threshold": 2.0, "read_file": "metrics/translation/data/french_failures.txt"}), 
    
        #(Translation, {"name": "spanish_french_transfer", "language": "French", "threshold": 0.9, "read_file": "metrics/translation/data/spanish_failures.txt"}),
        #(Translation, {"name": "spanish_dutch_transfer", "language": "Dutch", "threshold": 0.9, "read_file": "metrics/translation/data/spanish_failures.txt"}),
        #(Translation, {"name": "spanish_chinese_transfer", "language": "Mandarin Chinese", "threshold": 2.0, "read_file": "metrics/translation/data/spanish_failures.txt"}),
        #(Translation, {"name": "spanish_arabic_transfer", "language": "Arabic", "threshold": 2.0, "read_file": "metrics/translation/data/spanish_failures.txt"}),
        #(Translation, {"name": "spanish_korean_transfer", "language": "Korean", "threshold": 2.0, "read_file": "metrics/translation/data/spanish_failures.txt"}),
   
        #(Translation, {"name": "dutch_french_transfer", "language": "French", "threshold": 0.9, "read_file": "metrics/translation/data/dutch_failures.txt"}),
        #(Translation, {"name": "dutch_spanish_transfer", "language": "Spanish", "threshold": 0.9, "read_file": "metrics/translation/data/dutch_failures.txt"}),
        #(Translation, {"name": "dutch_arabic_transfer", "language": "Arabic", "threshold": 0.9, "read_file": "metrics/translation/data/dutch_failures.txt"}),
        #(Translation, {"name": "dutch_chinese_transfer", "language": "Mandarin Chinese", "threshold": 0.9, "read_file": "metrics/translation/data/dutch_failures.txt"}),
        #(Translation, {"name": "dutch_korean_transfer", "language": "Korean", "threshold": 0.9, "read_file": "metrics/translation/data/dutch_failures.txt"}),

        #(Translation, {"name": "chinese", "language": "Mandarin Chinese", "threshold": 2.0}),
        #(Translation, {"name": "chinese_spanish_transfer", "language": "Spanish", "threshold": 2.0, "read_file": "metrics/translation/data/chinese_failures.txt"}), 
        #(Translation, {"name": "chinese_french_transfer", "language": "French", "threshold": 2.0, "read_file": "metrics/translation/data/chinese_failures.txt"}),
        #(Translation, {"name": "chinese_dutch_transfer", "language": "Dutch", "threshold": 2.0, "read_file": "metrics/translation/data/chinese_failures.txt"}), 
        #(Translation, {"name": "chinese_arabic_transfer", "language": "Arabic", "threshold": 2.0, "read_file": "metrics/translation/data/chinese_failures.txt"}), 
        #(Translation, {"name": "chinese_chinese_transfer", "language": "Cantonese", "threshold": 2.0, "read_file": "metrics/translation/data/chinese_failures.txt"}),
        #(Translation, {"name": "chinese_korean_transfer", "language": "Korean", "threshold": 2.0, "read_file": "metrics/translation/data/chinese_failures.txt"}), 
    
        #(Translation, {"name": "arabic", "language": "Arabic", "threshold": 2.0}),
        #(Translation, {"name": "arabic_spanish_transfer", "language": "Spanish", "threshold": 2.0, "read_file": "metrics/translation/data/arabic_failures.txt"}), 
        #(Translation, {"name": "arabic_french_transfer", "language": "French", "threshold": 2.0, "read_file": "metrics/translation/data/arabic_failures.txt"}),
        #(Translation, {"name": "arabic_dutch_transfer", "language": "Dutch", "threshold": 2.0, "read_file": "metrics/translation/data/arabic_failures.txt"}), 
        #(Translation, {"name": "arabic_chinese_transfer", "language": "Mandarin Chinese", "threshold": 2.0, "read_file": "metrics/translation/data/arabic_failures.txt"}), 
        #(Translation, {"name": "arabic_korean_transfer", "language": "Korean", "threshold": 2.0, "read_file": "metrics/translation/data/arabic_failures.txt"}), 
    
        #(Translation, {"name": "korean", "language": "Korean", "threshold": 2.0}),
        #(Translation, {"name": "korean_french_transfer", "language": "French", "threshold": 2.0, "read_file": "metrics/translation/data/korean_failures.txt"}),
        #(Translation, {"name": "korean_spanish_transfer", "language": "Spanish", "threshold": 2.0, "read_file": "metrics/translation/data/korean_failures.txt"}),
        #(Translation, {"name": "korean_dutch_transfer", "language": "Dutch", "threshold": 2.0, "read_file": "metrics/translation/data/korean_failures.txt"}),
        #(Translation, {"name": "korean_arabic_transfer", "language": "Arabic", "threshold": 2.0, "read_file": "metrics/translation/data/korean_failures.txt"}),
        #(Translation, {"name": "korean_chinese_transfer", "language": "Mandarin Chinese", "threshold": 2.0, "read_file": "metrics/translation/data/korean_failures.txt"}),
    ])

    """ Studying randomness of Yes/No scoring """
    
    #for i in range(5):
    #    tasks.append((Translation, {"name": f"french_{i}", "language": "French", "threshold": 2.0}))
    #    tasks.append((Translation, {"name": f"french_dutch_transfer_{i}", "language": "Dutch", "threshold": 2.0, "read_file": f"metrics/translation/data/french_{i}_failures.txt"}))


if summarization:
    tasks.extend([
        (SummarizationScore, {"name": "news", "domain": ("news articles", news_scraper)}),
        (SummarizationScore, {"name": "short_stories", "domain": ("short stories", stories_scraper)}),
        
        #(SummarizationScore, {"name": "news_short_stories_transfer", "domain": ("short stories", stories_scraper), "read_file": "metrics/summarization/news_failures.txt"}),

        #(SummarizationScore, {"name": "short_stories_news_transfer", "domain": ("news articles", news_scraper), "read_file": "metrics/summarization/short_stories_failures.txt"}),
    
        #(SummarizationScore, {"name": "arxiv", "domain": ("technical scientific papers", arxiv_scraper)}),
        #(SummarizationScore, {"name": "arxiv_news_transfer", "domain": ("news articles", news_scraper), "read_file": "metrics/summarization/arxiv_failures.txt"}),
        #(SummarizationScore, {"name": "arxiv_short_stories_transfer", "domain": ("short stories", stories_scraper), "read_file": "metrics/summarization/arxiv_failures.txt"}),
        #(SummarizationScore, {"name": "arxiv_pubmed_transfer", "domain": ("medical scientific papers", pubmed_scraper), "read_file": "metrics/summarization/arxiv_failures.txt"}),
        
        #(SummarizationScore, {"name": "pubmed", "domain": ("medical scientific papers", pubmed_scraper)}),
    ])


all_metrics = []

for failure_mode in failure_modes:
    metrics = []

    for tup in tasks:
        task_type, params = tup

        if logger.check_finish(params["name"]):
            main_logger.print(f"Skipping job {params['name']}")
            continue
        
        logger.change_job(params["name"])
        main_logger.print(f"Starting task: {task_type} with params: {params}...")
        
        instance = task_type(failure_mode, total_examples, interacter, logger, **params)
        
        instance.gen_data()
        instance.pipeline()

        metrics.append(instance.extract_metrics())
        main_logger.print(metrics[-1])
        logger.finish()
    
    main_logger.print(metrics)
    all_metrics.append(metrics)

with open("metrics/aggregate.txt", 'w') as f:
    for i in range(len(failure_modes)):
        f.write(f"Failure mode: {failure_modes[i]}\nRates: {all_metrics[i]}\n")

