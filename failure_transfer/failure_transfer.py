from translation_pipeline import Translation
from information_retrieval_pipeline import InformationRetrieval
from summarization_score_pipeline import SummarizationScore
from style_gen_pipeline import StyleGeneration
from text_reorder_pipeline import TextReorder
from pos_pipeline import PartOfSpeech
from sentence_interweave_pipeline import SentenceInterweave
from sentence_half_pipeline import SentenceHalf
from domain_shift_pipeline import DomainShift
from completion_pipeline import Completion
from argument_pipeline import Argument
from ambiguity_pipeline import Ambiguity
from passive_pipeline import Passive
from sentence_succinct_pipeline import SentenceSuccinct
from pronoun_replacement_pipeline import PronounReplacement
from complex_replacement_pipeline import ComplexReplacement

from reasoning_pipeline import Reasoning
from logger import Logger
from interact_llama import InteractLLaMA
from interact_mistral import InteractMistral
from squad_generation import SquadSentenceDataset, SquadParagraphDataset
from ast import literal_eval

from scrape.scrape_stories import StoriesScrape

import re
import random

failure_modes = []

with open("failure_modes.txt", 'r') as f:
    for line in f:
        failure_modes.append(line.strip().replace('\n', ''))

domain_shift = False

''' Realistic Tasks '''
translation = True
translation_type = "paragraph"
summarization = True
completion = True

''' Synthetic Tasks '''
text_reordering = True
sentence_interweave = True
pronoun_replacement = True
complex_replacement = False

''' Extraneous Tasks '''
argument = False
ambiguity = False
passive = False
sentence_succinct = False
style_generation = False
pos = False
sentence_half = False   

logger = Logger()
main_logger = Logger()
main_logger.change_job("main")

#interacter = InteractLLaMA(logger)
interacter = InteractMistral(logger)

stories_scraper = StoriesScrape()
#news_scraper = NewsScrape()
#stories_scraper = StoriesScrape()
#arxiv_scraper = ArxivScrape()

sentences = []
tasks = []

total_examples = 10000
recover = True

paths = [
    #("data/domain_shift/squad_paragraphs_complex_examples.txt", "squad_paragraphs_complex", False),
    #("data/domain_shift/squad_paragraphs_disaster_examples.txt", "squad_paragraphs_disaster", True),

    (None, "squad_paragraphs", False),
    #(None, "short_stories", False),
    #("data/domain_shift/short_stories_medical_documents_examples.txt", "short-stories_med", False),
    #("data/domain_shift/short_stories_bulleted_examples.txt", "short-stories_bullet", True),
    #("data/domain_shift/short_stories_dialogue_examples.txt", "short-stories_dialogue", True),
]

if recover:
    input_domains = []
        
    for path in paths:
        if (path[0] == None):
            input_domain = []
            
            spd = SquadParagraphDataset('train')
            spd_gen = spd.paragraph_generator()  
        
            for i in range(2 * total_examples):
                try:
                    input_domain.append(next(spd_gen))      
                except:
                    break  
            
            input_domains.append((input_domain[:total_examples], path[1]))
            
            '''
            input_domain = stories_scraper.get_examples(total_examples)
            input_domains.append((input_domain, path[1]))
            '''
            
            main_logger.print(f"Recovering data; Length: {len(input_domain[:total_examples])}")
            continue
        
        input_domain = []

        if path[2]:
            with open(path[0], 'r') as file:
                lines = file.readlines()
                for i in range(0, len(lines), 4):
                    group = lines[i:i+4]
                    grouped_string = ''.join(group)
                    input_domain.append(grouped_string)
            
        else:
            with open(path[0], "r") as f:
                for line in f:
                    res = ""
                    
                    try:
                        tup = literal_eval(line)
                        res = tup[0]
                    except:
                        res = line
                    
                    input_domain.append(res)
    
        input_domain = input_domain[:total_examples]    
        main_logger.print(f"Recovering data; Length: {len(input_domain)}")
        
        input_domains.append((input_domain, path[1]))

else:
    input_domain = stories_scraper.get_examples(total_examples)

if domain_shift:
    complex_prefix = "Given a paragraph, please rewrite it so that it is more complex. Focus on making the grammar more complex, incorporating technical vocabulary, using analogies and metaphors, and expand upon ideas. Text: [\n"
    complex_suffix = "]\nRewrite the paragraph so that it is more complex. Be creative with how you make the paragraph more complex, and try to keep the meaning of the original paragraph intact. Format your output paragraph as: 'Output:...', and do not include any explanation."
    
    with open('prompts/disaster_report.txt', 'r') as f:
        disaster_prompt_prefix = f.read() + "\n"
    
    disaster_prompt_suffix = "]\n"
    
    bullet_prefix = "Given a paragraph, please rewrite it as a list of bullet points which capture the main plot points and characters. Paragraph: [\n"
    bullet_suffix = "]\nMinimize the number of bullet points. Format your bullet point output as: 'Output: ...', and do not include any explanation or additional dialogue."
    
    dialogue_prefix = "Given a paragraph, please rewrite it as a series of dialogue between all character interactions and spoken aspects. Ensure that the original narrative spirit of the paragraph remains intact in the dialogue. Paragraph: [\n"
    dialogue_suffix = "]\nYour output should be a series of dialogues, and do not add or delete narrative elements from the paragraph. Minimize the number of dialogues used without losing meaning in the original paragraph. Format your dialogue output as: 'Output: ...', and do not include any explanation."

    email_prefix = "Given a paragraph, please rewrite it as a series of emails exchanged between charaacters. Ensure that the original narrative spirit of the paragraph remains intact in the series of emails. Paragraph: [\n"
    email_suffix = "]\nMinimize the number of emails exchanged. All emails should be written in an urgent tone, and as if they were addressed to an executive employee or boss. Format your series of emails as: 'Output: ...', and do not include any explanation or additional dialogue."
    
    for input_domain in input_domains:
        pattern_prefix = "Given a paragraph, please rewrite it according to the instructions that will follow. Paragraph: [\n"
        pattern_suffix = "]\n. Instructions:\n"
        
        with open('failure_patterns.txt', 'r') as f:
            data = f.read()
            prompts = data.split('===')
        
            for i in range(len(prompts)):
                tasks.extend([
                    (DomainShift, {"initial_domain": input_domain[0], "input_domain": input_domain[1], "output_domain": f"failure_pattern_gen", "name": f"pattern_{i}", "domain_prefix": pattern_prefix, "domain_suffix": pattern_suffix + prompts[i]})
                ])
        
        tasks.extend([
            #(DomainShift, {"initial_domain": input_domain[0], "input_domain": input_domain[1], "output_domain": "complex", "name": "complex","domain_prefix": complex_prefix, "domain_suffix": complex_suffix}),

            #(DomainShift, {"initial_domain": input_domain[0], "input_domain": input_domain[1], "output_domain": "disaster", "name": "disaster_report", "domain_prefix": disaster_prompt_prefix, "domain_suffix": disaster_prompt_suffix}),
            
            # (DomainShift, {"initial_domain": input_domain[0], "input_domain": input_domain[1], "output_domain": "bulleted", "name": "bullet", "domain_prefix": bullet_prefix, "domain_suffix": bullet_suffix}),
            #(DomainShift, {"initial_domain": input_domain[0], "input_domain": input_domain[1], "output_domain": "dialogue", "name": "dialogue", "domain_prefix": dialogue_prefix, "domain_suffix": dialogue_suffix}),
            #(DomainShift, {"initial_domain": input_domain[0], "input_domain": input_domain[1], "output_domain": "email", "name": "email", "domain_prefix": email_prefix, "domain_suffix": email_suffix}),
        ])

if translation:

    if translation_type == "sentences":            
        ssd = SquadSentenceDataset('train')
        ssd_gen = ssd.sentence_generator()

        for i in range(2 * total_examples):
            cur_sentence = next(ssd_gen)
            
            if re.search(r'^[A-Za-z0-9 ,;!?]+$', cur_sentence):
                sentences.append(cur_sentence)
                           
        sentences = sentences[:total_examples]
                               
        tasks.extend([
            (Translation, {"initial_domain": sentences, "name": "french_sentence", "language": "French", "threshold": 0.9, "type": "sentence"}),
            (Translation, {"initial_domain": sentences, "name": "spanish_sentence", "language": "Spanish", "threshold": 0.9, "type": "sentence"}),
            (Translation, {"initial_domain": sentences, "name": "dutch_sentence", "language": "Dutch", "threshold": 0.9, "type": "sentence"}),
        ])
    
    else:                
        for input_domain in input_domains:            
            tasks.extend([
                #(Translation, {"initial_domain": input_domain[0], "name": input_domain[1] + "_french_paragraph", "language": "French", "threshold": 0.9, "type": "paragraph"}),
                (Translation, {"initial_domain": input_domain[0], "name": input_domain[1] + "_spanish_paragraph", "language": "Spanish", "threshold": 0.9, "type": "paragraph"}),
                #(Translation, {"initial_domain": input_domain[0], "name": input_domain[1] + "_dutch_paragraph", "language": "Dutch", "threshold": 0.9, "type": "paragraph"}),
            ])
    
if summarization:
    for input_domain in input_domains:
        tasks.extend([
            (SummarizationScore, {"initial_domain": input_domain[0], "name": input_domain[1]}),
            
            #(SummarizationScore, {"name": "summarization_translation_transfer", "read_file": "metrics/translation/data/translation_failures.txt"}),
            #(SummarizationScore, {"name": "short_translation_transfer", "read_file": "metrics/translation/data/translation_failures.txt"}),
            #(SummarizationScore, {"name": "arxiv_translation_transfer", "read_file": "metrics/translation/data/translation_failures.txt"}),

            #(SummarizationScore, {"name": "summarization_translation_baseline_transfer", "read_file": "metrics/translation/data/translation_baseline_failures.txt"}),
            #(SummarizationScore, {"name": "short_translation_baseline_transfer", "read_file": "metrics/translation/data/translation_baseline_failures.txt"}),
            #(SummarizationScore, {"name": "arxiv_translation_baseline_transfer", "read_file": "metrics/translation/data/translation_baseline_failures.txt"}),
            
            #(SummarizationScore, {"name": "arxiv_short_stories_transfer", "domain": ("short stories", stories_scraper), "read_file": "metrics/summarization/data/arxiv_failures.txt"}),
            #(SummarizationScore, {"name": "arxiv_pubmed_transfer", "domain": ("medical scientific papers", pubmed_scraper), "read_file": "metrics/summarization/data/arxiv_failures.txt"}),    
        ])

if style_generation:
    for input_domain in input_domains:
        tasks.extend([
            (StyleGeneration, {"initial_domain": input_domain[0], "name": input_domain[1] + "_present", "style": "present-tense text"}),
            (StyleGeneration, {"initial_domain": input_domain[0], "name": input_domain[1] + "_past", "style": "past-tense text"}),
            (StyleGeneration, {"initial_domain": input_domain[0], "name": input_domain[1] + "_future", "style": "future-tense text"}),
        ])    

if text_reordering:
    for input_domain in input_domains:
        tasks.extend([
            (TextReorder, {"initial_domain": input_domain[0], "name": input_domain[1]}),
        ])

if pos:
    for input_domain in input_domains:
        tasks.extend([
            (PartOfSpeech, {"initial_domain": input_domain[0], "name": input_domain[1] + "_noun", "pos_type": "noun"}),
            (PartOfSpeech, {"initial_domain": input_domain[0], "name": input_domain[1] + "_adj", "pos_type": "adjective"}),
            (PartOfSpeech, {"initial_domain": input_domain[0], "name": input_domain[1] + "_verb", "pos_type": "verb"}),
        ])
    
if sentence_interweave:
    for input_domain in input_domains:
        tasks.extend([
            (SentenceInterweave, {"initial_domain": input_domain[0], "name": input_domain[1]})
        ])
    
if sentence_half:
    for input_domain in input_domains:
        tasks.extend([
            (SentenceHalf, {"initial_domain": input_domain[0], "name": input_domain[1]})
        ])

if completion:
    for input_domain in input_domains:
        tasks.extend([
            (Completion, {"initial_domain": input_domain[0], "name": input_domain[1]})
        ])

if argument:
    for input_domain in input_domains:
        tasks.extend([
            (Argument, {"initial_domain": input_domain[0], "name": input_domain[1]})
        ])

if ambiguity:
    for input_domain in input_domains:
        tasks.extend([
            (Ambiguity, {"initial_domain": input_domain[0], "name": input_domain[1]})
        ])

if passive:
    for input_domain in input_domains:
        tasks.extend([
            (Passive, {"initial_domain": input_domain[0], "name": input_domain[1]})
        ])

if sentence_succinct:
    for input_domain in input_domains:
        tasks.extend([
            (SentenceSuccinct, {"initial_domain": input_domain[0], "name": input_domain[1]})
        ])

if pronoun_replacement:
    for input_domain in input_domains:
        tasks.extend([
            (PronounReplacement, {"initial_domain": input_domain[0], "name": input_domain[1]})
        ])
    
if complex_replacement:
    for input_domain in input_domains:
        tasks.extend([
            (ComplexReplacement, {"initial_domain": input_domain[0], "name": input_domain[1]})
        ])
    
tasks.sort(key = lambda t : len(t[1]["initial_domain"]))
    
debug = True
all_metrics = []

for failure_mode in failure_modes:
    metrics = []

    for tup in tasks:
        task_type, params = tup

        if logger.check_finish(params["name"]) and not debug:
            main_logger.print(f"Skipping job {params['name']}")
            continue
        
        logger.change_job(params["name"])
        main_logger.print(f"Starting task: {task_type} with name: {params['name']}...")
        
        instance = task_type(failure_mode, total_examples, interacter, logger, **params)
        
        instance.gen_data()
        instance.pipeline()
        
        print("Finished pipeline")

        metrics.append(instance.extract_metrics())
        main_logger.print(metrics[-1])
        logger.finish()
    
    main_logger.print(metrics)
    all_metrics.append(metrics)

with open("metrics/aggregate.txt", 'w') as f:
    for i in range(len(failure_modes)):
        f.write(f"Failure mode: {failure_modes[i]}\nRates: {all_metrics[i]}\n")

