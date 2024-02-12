from translation_pipeline import Translation
from information_retrieval_pipeline import InformationRetrieval
from summarization_score_pipeline import SummarizationScore
from reasoning_pipeline import Reasoning
from interact_llama import InteractLLaMA

from scrape.scrape_news import NewsScrape
from scrape.scrape_stories import StoriesScrape
from scrape.scrape_arxiv import ArxivScrape
from scrape.scrape_pubmed import PubmedScrape

#from scrape.scrape_amazon_reviews import AmazonReviewsScrape
#from scrape.scrape_law import LawScrape

failure_modes = []

with open("failure_modes.txt", 'r') as f:
    for line in f:
        failure_modes.append(line.strip().replace('\n', ''))

translation = True
summarization = True

assert((translation or summarization) == True)

if summarization:
    news_scraper = NewsScrape()
    stories_scraper = StoriesScrape()
    arxiv_scraper = ArxivScrape()
    pubmed_scraper = PubmedScrape()
    
    #amazon_reviews_scraper = AmazonReviewsScrape()
    #law_scraper = LawScrape()

tasks = []

if translation:
    #data_gen = SentenceScrape()
    data_gen = None

    tasks.extend([
        #(Translation, {"name": "french", "language": "French", "threshold": 2.0}),
        #(Translation, {"name": "french_spanish_transfer", "language": "Spanish", "threshold": 2.0, "read_file": "metrics/translation/french_failures.txt"}), 
        #(Translation, {"name": "french_dutch_transfer", "language": "Dutch", "threshold": 2.0, "read_file": "metrics/translation/french_failures.txt"}),
        #(Translation, {"name": "french_arabic_transfer", "language": "Arabic", "threshold": 2.0, "read_file": "metrics/translation/french_failures.txt"}),
        #(Translation, {"name": "french_chinese_transfer", "language": "Mandarin Chinese", "threshold": 2.0, "read_file": "metrics/translation/french_failures.txt"}),
        #(Translation, {"name": "french_korean_transfer", "language": "Korean", "threshold": 2.0, "read_file": "metrics/translation/french_failures.txt"}), 
    
        #(Translation, {"name": "spanish", "language": "Spanish", "threshold": 2.0}),
        #(Translation, {"name": "spanish_french_transfer", "language": "French", "threshold": 2.0, "read_file": "metrics/translation/spanish_failures.txt"}),
        #(Translation, {"name": "spanish_dutch_transfer", "language": "Dutch", "threshold": 2.0, "read_file": "metrics/translation/spanish_failures.txt"}),
        #(Translation, {"name": "spanish_chinese_transfer", "language": "Mandarin Chinese", "threshold": 2.0, "read_file": "metrics/translation/spanish_failures.txt"}),
        #(Translation, {"name": "spanish_arabic_transfer", "language": "Arabic", "threshold": 2.0, "read_file": "metrics/translation/spanish_failures.txt"}),
        #(Translation, {"name": "spanish_korean_transfer", "language": "Korean", "threshold": 2.0, "read_file": "metrics/translation/spanish_failures.txt"}),
   
        #(Translation, {"name": "dutch", "language": "Dutch", "threshold": 2.0}),
        #(Translation, {"name": "dutch_french_transfer", "language": "French", "threshold": 2.0, "read_file": "metrics/translation/dutch_failures.txt"}),
        #(Translation, {"name": "dutch_spanish_transfer", "language": "Spanish", "threshold": 2.0, "read_file": "metrics/translation/dutch_failures.txt"}),
        #(Translation, {"name": "dutch_arabic_transfer", "language": "Arabic", "threshold": 2.0, "read_file": "metrics/translation/dutch_failures.txt"}),
        #(Translation, {"name": "dutch_chinese_transfer", "language": "Mandarin Chinese", "threshold": 2.0, "read_file": "metrics/translation/dutch_failures.txt"}),
        #(Translation, {"name": "dutch_korean_transfer", "language": "Korean", "threshold": 2.0, "read_file": "metrics/translation/dutch_failures.txt"}),

        #(Translation, {"name": "chinese", "language": "Mandarin Chinese", "threshold": 2.0}),
        #(Translation, {"name": "chinese_spanish_transfer", "language": "Spanish", "threshold": 2.0, "read_file": "metrics/translation/chinese_failures.txt"}), 
        #(Translation, {"name": "chinese_french_transfer", "language": "French", "threshold": 2.0, "read_file": "metrics/translation/chinese_failures.txt"}),
        #(Translation, {"name": "chinese_dutch_transfer", "language": "Dutch", "threshold": 2.0, "read_file": "metrics/translation/chinese_failures.txt"}), 
        #(Translation, {"name": "chinese_arabic_transfer", "language": "Arabic", "threshold": 2.0, "read_file": "metrics/translation/chinese_failures.txt"}), 
        #(Translation, {"name": "chinese_chinese_transfer", "language": "Cantonese", "threshold": 2.0, "read_file": "metrics/translation/chinese_failures.txt"}),
        #(Translation, {"name": "chinese_korean_transfer", "language": "Korean", "threshold": 2.0, "read_file": "metrics/translation/chinese_failures.txt"}), 
    
        #(Translation, {"name": "arabic", "language": "Arabic", "threshold": 2.0}),
        #(Translation, {"name": "arabic_spanish_transfer", "language": "Spanish", "threshold": 2.0, "read_file": "metrics/translation/arabic_failures.txt"}), 
        #(Translation, {"name": "arabic_french_transfer", "language": "French", "threshold": 2.0, "read_file": "metrics/translation/arabic_failures.txt"}),
        #(Translation, {"name": "arabic_dutch_transfer", "language": "Dutch", "threshold": 2.0, "read_file": "metrics/translation/arabic_failures.txt"}), 
        #(Translation, {"name": "arabic_chinese_transfer", "language": "Mandarin Chinese", "threshold": 2.0, "read_file": "metrics/translation/arabic_failures.txt"}), 
        #(Translation, {"name": "arabic_korean_transfer", "language": "Korean", "threshold": 2.0, "read_file": "metrics/translation/arabic_failures.txt"}), 
    
        #(Translation, {"name": "korean", "language": "Korean", "threshold": 2.0}),
        #(Translation, {"name": "korean_french_transfer", "language": "French", "threshold": 2.0, "read_file": "metrics/translation/korean_failures.txt"}),
        #(Translation, {"name": "korean_spanish_transfer", "language": "Spanish", "threshold": 2.0, "read_file": "metrics/translation/korean_failures.txt"}),
        #(Translation, {"name": "korean_dutch_transfer", "language": "Dutch", "threshold": 2.0, "read_file": "metrics/translation/korean_failures.txt"}),
        #(Translation, {"name": "korean_arabic_transfer", "language": "Arabic", "threshold": 2.0, "read_file": "metrics/translation/korean_failures.txt"}),
        #(Translation, {"name": "korean_chinese_transfer", "language": "Mandarin Chinese", "threshold": 2.0, "read_file": "metrics/translation/korean_failures.txt"}),
    ])

    """ Studying randomness of Yes/No scoring """
    for i in range(5):
        tasks.append((Translation, {"name": "french", "language": "French", "threshold": 2.0}))
        tasks.append((Translation, {"name": f"french_dutch_transfer_{i}", "language": "Dutch", "threshold": 2.0, "read_file": "metrics/translation/french_failures.txt"}))


if summarization:
    tasks.extend([
        #(SummarizationScore, {"name": "news", "domain": ("news articles", news_scraper)}),
        #(SummarizationScore, {"name": "news_short_stories_transfer", "domain": ("short stories", stories_scraper), "read_file": "metrics/summarization/news_failures.txt"}),

        #(SummarizationScore, {"name": "short_stories", "domain": ("short stories", stories_scraper)}),
        #(SummarizationScore, {"name": "short_stories_news_transfer", "domain": ("news articles", news_scraper), "read_file": "metrics/summarization/short_stories_failures.txt"}),
    
        (SummarizationScore, {"name": "arxiv", "domain": ("technical scientific papers", arxiv_scraper)}),
        (SummarizationScore, {"name": "arxiv_news_transfer", "domain": ("news articles", news_scraper), "read_file": "metrics/summarization/arxiv_failures.txt"}),
        (SummarizationScore, {"name": "arxiv_short_stories_transfer", "domain": ("short stories", stories_scraper), "read_file": "metrics/summarization/arxiv_failures.txt"}),
        (SummarizationScore, {"name": "arxiv_pubmed_transfer", "domain": ("medical scientific papers", pubmed_scraper), "read_file": "metrics/summarization/arxiv_failures.txt"}),
        
        #(SummarizationScore, {"name": "pubmed", "domain": ("medical scientific papers", pubmed_scraper)}),

        #(SummarizationScore, {"name": "amazon_reviews", "domain": ("amazon reviews", amazon_reviews_scraper)}),
        #(SummarizationScore, {"name": "law", "domain": ("legal documents", law_scraper)}),
    ])


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

