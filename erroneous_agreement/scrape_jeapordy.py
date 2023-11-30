import json

default_file_name = 'data/JEOPARDY_QUESTIONS1.json'

class JeapordyParser:
    def __init__(self, file_name = default_file_name):
        self.file_name = file_name
        
    def get_questions(self):
        with open(self.data) as json_data:
            data = json.loads(json_data.read())

        questions = []
        for element in data:
            include = True
            question = element["question"]

            # Ignore files
            if "http://" in question or "https://" in question:
                include = False

            if include:
                questions.append(question)
        
        return (questions)
