from scrape_tinystories import TinyStoriesParser
from scrape_news import NewsParser
from scrape_file import FileParser
from scrape_hf_dataset import HFParser
from interact_llama import InteractLLaMA
import random, math
import re
import os

parser = TinyStoriesParser()

trap_story = ["Jax is a black cat.", "Jax's owner is a small boy named Emerson.", "When Jax gets hunry, Emerson gives Jax some milk", 
              "Drinking milk makes Jax sleepy.", "Jax's favorite spot to stretch and yawn is outside in the bright sun", 
              "Emerson is thinking of taking Jax to the vet", "Run Jax run!"]

trap_keywords = ["Jax", "black cat", "Emerson", "stretch and yawn", "outside in the bright sun", "vet"]

def inject_trap_story(story):
    inject_threshold = 0.15
    acceptable_len = int(inject_threshold * len(story))
    
    sum_len = 0
    idx = 0
    while idx < len(trap_story):
        sum_len += len(trap_story[idx])
        
        if sum_len > acceptable_len:
            idx -= 1
            break

        idx += 1
    
    story_sentences = re.split(r'([.!?])+', story)
    sentence_inject_idxs = random.sample(range(0, len(story_sentences)), idx)
    sentence_inject_idxs.sort()

    new_story = ""
    trap_idx = 0

    for i in range(len(story_sentences)):
        new_story += story_sentences[i]

        if len(sentence_inject_idxs) > 0 and i in sentence_inject_idxs:
            new_story += (". " + trap_story[trap_idx])
            trap_idx += 1

    new_story = new_story.replace("..", ".")
    return (new_story)


def extract_questions(list_of_stories):
    prefix_q = "I will provide you with some sample text. Please summarize this text as succinctly as possible without losing meaning. Please focus on relevant details, and do not include any other text apart from the summary. "

    new_questions = []
    for story in list_of_stories:
        story = story.strip().replace('\n', '')
        story = inject_trap_story(story)    
        new_questions.append(prefix_q + story)

    return (new_questions)
    
def extract_answers(response):
    return ([response])

stories = parser.get_stories()
random.shuffle(stories)

# For iterative failures generated
word_limit = 1000

stories = list(filter(lambda x : x != "" and len(x.split(' ')) <= word_limit, stories))
short_stories = stories[:1000]
print("Number of possible stories: ", len(stories))

interacter = InteractLLaMA()
failures = []

questions = extract_questions(short_stories)
answers = interacter.answer_questions(questions, extract_answers)

for i in range(len(answers)):
    for word in trap_keywords:
        if word in answers[i]:
            failures.append((questions[i], answers[i]))
            break

output_directory = 'summarize_failure_output/stories'

try:
    os.mkdir(output_directory)
except OSError as error:
    print(f"Could not create directory along path {output_directory}; {error}\n")

with open(output_directory + '/all_failures.txt', 'w') as f:
    f.write(f"Failure rate: {len(failures) / max(1, len(answers))}\n")
    
    for failure in failures:
        f.write(f"Story: {failure[0]}\nResponse: {failure[1]}\n")
