from scrape_tinystories import TinyStoriesParser
from scrape_news import NewsParser
from scrape_file import FileParser
from scrape_hf_dataset import HFParser
import random, math
import re
import gc
import os

parser = TinyStoriesParser()
stories = parser.get_stories()
stories = list(filter(lambda x : x != "" and len(x.split(' ')) <= 60, stories))
random.shuffle(stories)
print("Number of possible stories: ", len(stories))

num_examples_rand = 100
with open('random_classification_prompt.txt', 'w') as f:
    f.write("[")

    for story in stories[:num_examples_rand]:
        story = story.replace("\n", "")
        f.write(f"({story})\n")
    
    f.write("]")
