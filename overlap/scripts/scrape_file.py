import os

class FileParser:
    def __init__(self, path, parse_func):
        self.path = path
        self.parse = parse_func

    def get_stories(self):
        stories = []
        
        with open(self.path) as f:
            for line in f:
                if line == '\n':
                    continue
        
                line = self.parse(line)
                stories.append(line)
        
        return (stories)

if __name__ == "__main__":
    sample_parser = FileParser('../data/stories_to_news.txt', lambda line : line.replace('"', '').strip())
    stories = sample_parser.get_stories()    
    
    print(stories)

