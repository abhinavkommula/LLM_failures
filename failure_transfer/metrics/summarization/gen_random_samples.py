import os
import random

from ast import literal_eval

cur_dir = './data/'

def form(fl):
    return float('{:,.3f}'.format(fl))

for filename in os.listdir(cur_dir):
    file_path = os.path.join(cur_dir, filename)

    if os.path.isfile(file_path):
        words = filename.split('_')

        if words[-1] == 'precision.txt':
            print("Examining file_path:", file_path)
                
            total_examples = []
            num_examples = 20

            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    tup = literal_eval(line)
                    
                    total_examples.append(tup)

            random.shuffle(total_examples)
            rand_examples = total_examples[:num_examples]

            print(len(rand_examples))

            for i in range(len(rand_examples)):
                tup = rand_examples[i]

                print(f"{i}: Original: {tup[0]}\nEntities: {tup[2]}\nSummary: {tup[1]}\nEntities: {tup[3]}\n")
                print("=================================")

            weak = [int(x) for x in input("Type list of indices that are weak: ").split()]
            weak_stats = []
            strong_stats = []

            for i in range(len(rand_examples)):
                tup = rand_examples[i]

                if i in weak:
                    weak_stats.append((i, form(tup[4]), form(tup[5])))
                else:
                    strong_stats.append((i, form(tup[4]), form(tup[5])))
            
            print("Weak example stats:")
            for el in weak_stats:
                print(el, end = " ")
            
            print("\n=================================")
            print("Strong example stats:")
            for el in strong_stats:
                print(el, end = " ")

            print("\n=================================")
            input("Press Enter to continue...")
