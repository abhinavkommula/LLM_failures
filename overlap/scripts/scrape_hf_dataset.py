from datasets import load_dataset

class HFParser:
    def __init__(self, dataset_name, parse_func):
        self.dataset = load_dataset(dataset_name)
        self.parse = parse_func

    def get_stories(self):
        stories = []
        for card in self.dataset['train']:
            stories.append(self.parse(card['content']))

        return (stories)

if __name__ == "__main__":
    def poetry_parser(x):
        poem = x.strip()

        if len(poem.split(' ')) > 200:
            poem = ""

        return (poem)

    sample_parser = HFParser("merve/poetry", poetry_parser)
    stories = sample_parser.get_stories()
    
    print(len(stories))
