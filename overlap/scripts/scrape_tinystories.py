from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories")

class TinyStoriesParser:
	def __init__(self):
		pass

	def get_stories(self):
		stories = []
		for card in dataset['train']:
			stories.append(card['text'])
        
		return (stories)

if __name__ == "__main__":
	sample_parser = TinyStoriesParser()
	stories = sample_parser.get_stories()
        
	print(stories)
