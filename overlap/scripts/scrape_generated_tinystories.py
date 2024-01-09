import os

class AdaptiveStoriesParser:
    def __init__(self, path):
        self.path = path

    def get_stories(self):
        stories = []
        
        with open(self.path) as f:
            for line in f:
                if line == '\n':
                    continue
        
                line = line.replace('"', '').replace('(', '').replace(')', '')
                stories.append(line)
        
        return (stories)

if __name__ == "__main__":
	sample_parser = AdaptiveStoriesParser()
	stories = sample_parser.get_stories()
        
	print(stories)
