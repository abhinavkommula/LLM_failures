from datasets import load_dataset

dataset = load_dataset("multi_news")

class NewsParser:
	def __init__(self):
		pass

	def get_stories(self):
		stories = []
		for card in dataset['train']:
			stories.append(card['summary'])
        
		return (stories)

if __name__ == "__main__":
	sample_parser = NewsParser()
	stories = sample_parser.get_stories()
        
	print(stories)
