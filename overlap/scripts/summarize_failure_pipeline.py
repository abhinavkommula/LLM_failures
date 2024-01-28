from scrape_tinystories import TinyStoriesParser
from scrape_news import NewsParser
from scrape_file import FileParser
from scrape_hf_dataset import HFParser
from interact_llama import InteractLLaMA
import random, math
import re
import os

#parser = TinyStoriesParser()
parser = NewsParser()

''' Level #1: Irrelevant Facts '''
facts = ["The all pairs shortest path algorithm is NP-complete", "A group of flamingos is called a flamboyance", "Octopuses have three hearts",
         "Bananas are berries, but strawberries are not", "A jiffy is a unit of time for 1/100th of a second"]

facts_keywords = ["NP-complete", "shortest path", "algorithm", "flamingos", "flamboyance", "octopuses", "three hearts", 
                  "bananas", "berries", "strawberries", "jiffy", "1/100th", "unit of time" ]

''' Level #2: Sequential Facts '''
sequentialfacts = ["Monarch butterflies lay their eggs on milkweed leaves", "The eggs hatch into larvae, known as caterpillars", 
         "Caterpillars feed only on milkweed, making them toxic to predators", "After feeding, the caterpillar forms a chrysalis" ]

sequentialfacts_keywords = ["Monarch Buterflies", "Eggs", "Milkweed", "Larvae", "Caterpillars", "Toxic", "Chrysalis" ]


''' Level #3: Mini Story '''
ministory = ["Jax is a black cat.", "Jax's owner is a small boy named Emerson.", "When Jax gets hunry, Emerson gives Jax some milk", 
              "Drinking milk makes Jax sleepy.", "Jax's favorite spot to stretch and yawn is outside in the bright sun", 
              "Emerson is thinking of taking Jax to the vet", "Run Jax run!"]

ministory_keywords = ["Jax", "black cat", "Emerson", "stretch and yawn", "outside in the bright sun", "vet"]

''' Level #0: No Change '''
nochange = []
nochange_keywords = []

gradation_injection_triggers = [(facts, facts_keywords), (sequentialfacts, sequentialfacts_keywords), (ministory, ministory_keywords), (nochange, nochange_keywords)]
gradation_failure_filenames = ["irrelevant_facts.txt", "sequential_facts.txt", "mini_story.txt", "no_change.txt"]
gradation_nonfailure_filenames = ["irrelevant_facts_nofail.txt", "sequential_facts_nofail.txt", "mini_story_nofail.txt", "no_change_nofail.txt"]

def inject_trigger(story, trigger_sentences):
    inject_threshold = 0.1
    acceptable_len = int(inject_threshold * len(story))
    
    sum_len = 0
    idx = 0
    while idx < len(trigger_sentences):
        sum_len += len(trigger_sentences[idx])
        idx += 1
        
        if sum_len > acceptable_len:
            idx -= 1
            break
    
    story_sentences = re.split(r'([.!?])+', story)
    idx = min(idx, len(story_sentences))

    sentence_inject_idxs = random.sample(range(0, len(story_sentences)), idx)
    sentence_inject_idxs.sort()

    new_story = ""
    trigger_idx = 0

    for i in range(len(story_sentences)):
        new_story += story_sentences[i]

        if len(sentence_inject_idxs) > 0 and i in sentence_inject_idxs:
            new_story += (". " + trigger_sentences[trigger_idx] + ".")
            trigger_idx += 1

    new_story = new_story.replace("..", ".")
    return (new_story)


def extract_questions(list_of_stories, trigger_sentences):
    prefix_q = "I will provide you with some sample text. Please summarize this text using at most 5 sentences. Please focus on relevant details. Do not respond with more than 5 sentences. "

    questions_injected = []
    questions = []

    for story in list_of_stories:
        story_injected = inject_trigger(story, trigger_sentences)    
        
        questions_injected.append(prefix_q + story_injected)
        questions.append(prefix_q + story)

    return (questions, questions_injected)
    
def extract_answers(response):
    return ([response])

stories = parser.get_stories()
random.shuffle(stories)

# For iterative failures generated
word_limit = 1000

stories = list(filter(lambda x : x != "" and len(x.split(' ')) <= word_limit, stories))
short_stories = stories[:1000]
short_stories = [s.strip().replace('\n', '') for s in short_stories]
print("Number of possible stories: ", len(stories))

output_directory = 'summarize_failure_output/news'

try:
    os.mkdir(output_directory)
except OSError as error:
    print(f"Could not create directory along path {output_directory}; {error}\n")

interacter = InteractLLaMA()
failures = []
nonfailures = []

for iteration in range(len(gradation_injection_triggers)):
    level = gradation_injection_triggers[iteration]

    trigger_sentences = level[0]
    trigger_keywords = level[1]

    level_failures = []
    level_nonfailures = []

    questions, questions_injected = extract_questions(short_stories, trigger_sentences)
    answers = interacter.answer_questions(questions, extract_answers)
    answers_injected = interacter.answer_questions(questions_injected, extract_answers)

    for i in range(len(answers_injected)):
        flag = False

        for word in trigger_keywords:
            if word.lower() in answers_injected[i].lower():
                flag = True
                break

        if flag or len(trigger_keywords) == 0:
            level_failures.append((short_stories[i], answers[i], questions_injected[i], answers_injected[i]))
        else:
            level_nonfailures.append((short_stories[i], answers[i], questions_injected[i], answers_injected[i]))
    
    failures.append(level_failures)
    nonfailures.append(level_nonfailures)

    with open(output_directory + '/' + gradation_failure_filenames[iteration], 'w') as f:
        f.write(f"Failure rate: {len(failures[iteration]) / max(1, len(short_stories))}\n")
    
        for failure in failures[iteration][:500]:
            f.write(f"Original Story: {failure[0]}\nOriginal Summary: {failure[1]}\nInjected Story: {failure[2]}\nInjected Summary: {failure[3]}\n")
    
    with open(output_directory + '/' + gradation_nonfailure_filenames[iteration], 'w') as f:
        for nonfailure in nonfailures[iteration][:500]:
            f.write(f"Original Story: {nonfailure[0]}\nOriginal Summary: {nonfailure[1]}\nInjected Story: {nonfailure[2]}\nInjected Summary: {nonfailure[3]}\n")
