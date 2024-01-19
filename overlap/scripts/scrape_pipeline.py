from scrape_tinystories import TinyStoriesParser
from scrape_news import NewsParser
from scrape_file import FileParser
from scrape_hf_dataset import HFParser
from interact_llama import InteractLLaMA
import random, math
import re
import os

#parser = NewsParser()
#parser = TinyStoriesParser()

parser = FileParser('generation_output/stories_indomain_all_failures.txt', lambda line : line.replace('"', '').strip())

'''
def poetry_parser(x):
    poem = x.strip()

    if len(poem.split(' ')) > 200:
        poem = ""

    return (poem)

parser = HFParser("merve/poetry", poetry_parser)
'''

'''
Verify overlap amongst pairs of opposite entites in dataset
'''

list_1 = ["important entities", "simple themes",  "physical actions", "male entities", "positive emotions"]
list_2 = ["irrelevant entities", "complex themes", "mental actions", "female entities", "negative emotions"]

'''
list_1 = ["protagonists", "old characters", "male characters", "indoor settings", "physical actions", "human characters", "positive emotions"]
list_2 = ["antagonists", "young characters", "female characters", "outdoor settings", "mental actions", "animal characters", "negative emotions"]
'''

def extract_questions(list_of_stories, search_text):
    prefix_q = "I will provide you with some sample text. From this text, please provide a list of all examples of " + search_text + ". Limit each answer in the list to 1 to 2 words. When writing your answer, generate it as: '[List: 1., 2., 3., ...]'"

    new_questions = []
    for story in list_of_stories:
        story = story.strip()
        new_questions.append(prefix_q + story)

    return (new_questions)
    
def extract_answers(response):
    items = re.split(r'\d+\.\s*', response)
    truncate_to_two_words = lambda x : ' '.join(x.split(' ')[:5])
    return ([truncate_to_two_words(item.strip()) for item in items if item])


stories = parser.get_stories()
random.shuffle(stories)

# For initial scraping
word_limit = 60

# For iterative failures generated
# word_limit = 1000

stories = list(filter(lambda x : x != "" and len(x.split(' ')) <= word_limit, stories))
short_stories = stories[:1000]
print("Number of possible stories: ", len(stories))

interacter = InteractLLaMA()

# Indexed by failure-test pair
overlaps = {}

total_errors = [0 for el in list_1]
total_questions = [0 for el in list_1]

for i in range(len(list_1)):
    overlaps[list_1[i]] = []

    questions_1 = extract_questions(short_stories, list_1[i])
    answers_1   = interacter.answer_questions(questions_1, extract_answers) 

    questions_2 = extract_questions(short_stories, list_2[i])
    answers_2   = interacter.answer_questions(questions_2, extract_answers)

    for a_idx in range(len(answers_1)):
        set1 = set(answers_1[a_idx])
        set2 = set(answers_2[a_idx])
        overlap = set1.intersection(set2)

        if overlap:
            total_errors[i] += 1 

            overlap_ratio = len(overlap) / (len(set1) + len(set2) - len(overlap))
            overlaps[list_1[i]].append((list_1[i], list_2[i], answers_1[a_idx], answers_2[a_idx], short_stories[a_idx], overlap_ratio))

        total_questions[i] += 1

    overlaps[list_1[i]].sort(key = lambda tup : -tup[5])

'''
Output file structure:

{language domain}_{domain type}_{failure test}_{?random}_{iteration #}
    /all_overlaps.txt
    /classification_prompt.txt
    /statistics.txt
'''

output_directory = 'scrape_output/stories_indomain_all_2'

try:
    os.mkdir(output_directory)
except OSError as error:
    print(f"Could not create directory along path {output_directory}; {error}\n")

with open(output_directory + '/all_overlaps.txt', 'w') as f:
    for failure_test in list_1:
        for o in overlaps[failure_test]:
            try:
                f.write(f"Story: {o[4]}\nEntity {o[0]}: {o[2]}\nEntity {o[1]}: {o[3]}\nOverlap Ratio: {o[5]}\n")
            except:
                continue

with open(output_directory + '/statistics.txt', 'w') as f:
    for i in range(len(list_1)):
        error_rate = round((total_errors[i] * 1.0) / total_questions[i], 2)
        binomial_error = round(1.96 * math.sqrt((error_rate * (1 - error_rate)) / total_questions[i]), 2)

        f.write(f"{list_1[i]}/{list_2[i]} Error rate: {error_rate}, Confidence interval: [{error_rate - binomial_error}, {error_rate + binomial_error}]\n")


num_examples_per_failure_test = 100
for i in range(len(list_1)):
    with open(output_directory + "/classification_prompt_" + list_1[i].replace(' ', '_') + "_" + list_2[i].replace(' ', '_') + ".txt", 'w') as f:
        
        f.write("[")

        for o in overlaps[list_1[i]][:num_examples_per_failure_test]:
            cur_story = o[4].replace("\n", "")
            f.write(f"({cur_story})\n")

        f.write("]")
