import json

default_file_name = 'alpaca_data.json'

class AlpacaParser:
    def __init__(self, file_name = default_file_name):
        self.file_name = file_name
        
    def get_questions(self):
        with open(self.data) as json_data:
            data = json.loads(json_data.read())

        questions = []
        for element in data:
            include = True
            question = element["instruction"]
            
            if len(element["input"]) > 0:
                question += (". " + element["input"])
            
            if include:
                questions.append(question)
        
        return (questions)
