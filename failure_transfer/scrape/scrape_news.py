from datasets import load_dataset
import random

dataset = load_dataset("cnn_dailymail", '3.0.0')

class NewsScrape:
    def __init__(self):
        self.gen_examples()

    def gen_examples(self):
        self.examples = []
        for card in dataset['train']:
            if len(card['article']) < 3000:
                self.examples.append(card['article'].replace('\n', ''))

    def get_random(self):
        return (random.choice(self.examples))

    def get_examples(self, num):
        return (self.examples[:min(len(self.examples), num)])

if __name__ == "__main__":
    sample_parser = NewsScrape()
    rand_example = sample_parser.get_random()
    two_examples = sample_parser.get_examples(2)

    print("Rand example:", rand_example)
    print("Two examples:", two_examples)
