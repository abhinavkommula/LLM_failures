from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
from collections import defaultdict, Counter
import numpy as np

if False:
    nltk.download('punkt')

class SentenceDataset:
    def __init__(self):
        pass

    def __len__(self):
        return len(self.sentences)

    def sentence_generator(self):
        for sentence in self.sentences:
            yield sentence

class SquadSentenceDataset(SentenceDataset):
    def __init__(self, split):
        squad_data = load_dataset('squad')
        train_data = squad_data[split]
        self.sentences = []
        passages_so_far = set()
        
        for i in range(len(train_data)):
            elem = train_data[i]
            context = elem['context']
            
            if context in passages_so_far:
                continue                                                                        

            passages_so_far.add(context)
            # Split context into sentences
            passage_sentences = sent_tokenize(context)
            self.sentences.extend(passage_sentences) 
