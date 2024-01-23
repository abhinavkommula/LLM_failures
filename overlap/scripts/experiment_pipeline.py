from interact_llama import InteractLLaMA
import random, math
import re
import os
import sys

def extract_answers(response):
    return ([response])

print("Loading interacter...")

interacter = InteractLLaMA()
question_number = 1

print("Finished loading interacter")

print(f"Question #{question_number}:\n")
question_number += 1

for question in sys.stdin:
    print(interacter.answer_questions([question], extract_answers)) 
    print(f"Question #{question_number}:\n")
    question_number += 1

