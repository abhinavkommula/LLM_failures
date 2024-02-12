from datasets import load_dataset
import random

dataset = load_dataset("amazon_reviews_multi")

class AmazonReviewsScrape:
    def __init__(self):
        self.gen_examples()

    def gen_examples(self):
        self.examples = []

        for card in dataset['train']:
            if len(card['review_body']) < 3000 and card['language'] == "en":
                self.examples.append(card['review_body'].replace('\n', ''))

    def get_random(self):
        return (random.choice(self.examples))

    def get_examples(self, num):
        return (self.examples[:min(len(self.examples), num)])

if __name__ == "__main__":
    sample_parser = AmazonReviewsScrape()
    rand_example = sample_parser.get_random()
    five_examples = sample_parser.get_examples(5)

    print("Rand example:", rand_example)
    print("Five examples:", five_examples)
