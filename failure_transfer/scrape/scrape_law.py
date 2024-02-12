from datasets import load_dataset
import random

dataset = load_dataset("pile-of-law/pile-of-law", "atticus_contracts")

class LawScrape:
    def __init__(self):
        self.gen_examples()

    def gen_examples(self):
        self.examples = []

        for card in dataset['train']:
            print(card)
            data = card['atticus_contracts'].replace('\n', '')
            rand_idx = random.randint(0, len(data) - 3000)

            self.examples.append(data[rand_idx : rand_idx + 3000])

    def get_random(self):
        return (random.choice(self.examples))

    def get_examples(self, num):
        return (self.examples[:min(len(self.examples), num)])

if __name__ == "__main__":
    sample_parser = LawScrape()
    rand_example = sample_parser.get_random()
    five_examples = sample_parser.get_examples(5)

    print("Rand example:", rand_example)
