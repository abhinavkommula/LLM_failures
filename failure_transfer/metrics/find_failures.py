from ast import literal_eval

file_name = "./summarization/data/squad_paragraphs_complex_failures.txt"

def skip_example(tup):
    if "Error" in tup[0] or "OpenAI" in tup[0]:
        return True
    
    if len(tup[1]) < 20 or tup[1][-1] != "."    :
        return True
    
    ''' Translation 
    if len(tup[0]) / len(tup[1]) >= 2.0:
        print(len(tup[0]), len(tup[1]))
        print(tup[0])
        print(tup[1])
        return True
    '''
    
    ''' Summarization '''
    
    return False

nfailures = 50
nnonfailures = 25
failures = []
nonfailures = []

with open(file_name, 'r') as f:
    for line in f:
        tup = literal_eval(line)
        
        if skip_example(tup):
            continue
        
        if (tup[2] == 0):
            failures.append(tup)
        else:
            nonfailures.append(tup)
            
failures.sort(key = lambda t : len(t[0]))
nonfailures.sort(key = lambda t : len(t[0]))
            
failures = failures[10:10+nfailures]
nonfailures = nonfailures[10:10+nnonfailures]

            
with open("prompt.txt", "w") as f:
    f.write("The following is a list of input paragraphs, along with a language model’s output, that have been flagged as failures. In each instance, the paragraph was translated by the LLM model to a language, then translated from that language back to English, and compared to the original input. Failures are identified when the back-translated output doesn’t match the input. Pay attention to both lists, in particular patterns in failures that do not appear in nonfailures.")
    
    f.write("\nFailures: [\n")
    
    for failure in failures:
        str1 = failure[0].replace('\n', '')
        str2 = failure[1].replace('\n', '')
        
        f.write(f"Failure: {str1}\n")
    
    f.write("\n]\nNonfailures: [\n")
    
    for nonfailure in nonfailures:
        str1 = nonfailure[0].replace('\n', '')
        str2 = nonfailure[1].replace('\n', '')
        
        f.write(f"Nonfailure: {str1}\n")
    
    f.write("\n]\n")
    f.write("WHat are novel patterns you notice amongst the inputs in the failures that likely lead to failures in the outputs, which do not appear in the nonfailures list? Focus on patterns related to translation. Focus on errors you notice in the falures themselves, and do not make any assumptions based on the fact that this is a translation task. Identify the most common patterns amongst failures, along with a detailed explanation of what the faiure is, specific examples of where it appears, and how to recreate this pattern.")