from scrape_tinystories import TinyStoriesParser
from scrape_news import NewsParser
from scrape_file import FileParser
from scrape_hf_dataset import HFParser
from interact_llama import InteractLLaMA
import random, math
import re
import os

parser = TinyStoriesParser()

def extract_questions(list_of_stories):
    prefix_q = "I will provide you with some sample text. Please summarize this text as succinctly as possible in 1 sentence. Please focus on relevant details. Only respond with 1 sentence. "

    new_questions = []
    for story in list_of_stories:
        story = story.strip()
        new_questions.append(prefix_q + story)

    return (new_questions)
    
def extract_answers(response):
    return ([response])

stories = parser.get_stories()
random.shuffle(stories)

# For initial scraping
# word_limit = 60

word_limit = 1000

stories = list(filter(lambda x : x != "" and len(x.split(' ')) <= word_limit, stories))
short_stories = stories[:500]
short_stories = [s.replace('\n', '').strip() for s in short_stories]
print("Number of possible summary stories: ", len(stories))

interacter = InteractLLaMA()

questions = extract_questions(short_stories)
answers   = interacter.answer_questions(questions, extract_answers) 

output_directory = 'summary_output/'

try:
    os.mkdir(output_directory)
except OSError as error:
    print(f"Could not create directory along path {output_directory}; {error}\n")

with open(output_directory + 'stories.txt', 'w') as f:
    for a_idx in range(len(answers)):
        f.write(f"(Story: {short_stories[a_idx]}\nSummary: {answers[a_idx]})\n")
