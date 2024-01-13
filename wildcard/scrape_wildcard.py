from datasets import load_dataset

dataset = load_dataset("allenai/WildChat")

class WildcardParser:
	def __init__(self):
		pass

	def get_prompt(self, filter_toxic = False, filter_nonenglish = False,
						 filter_redacted = False ):

		for card in dataset['train']:
			conversations = card['conversation']
			prompt_response = []
			
			assert(len(card['conversation']) == 2 * card['turn'])

			for turn in range(card['turn']):
				cur_prompt = card['conversation'][2 * turn]
				cur_response = card['conversation'][2 * turn + 1]

				if filter_toxic and (cur_prompt['toxic'] or cur_response['toxic']):
					continue
				
				elif filter_nonenglish and (cur_prompt['language'].lower() != "english" or
											cur_response['language'].lower() != "english"):
					continue
				
				elif filter_redacted and (cur_prompt['redacted'] or cur_response['redacted']):
					continue

				prompt_response.append((cur_prompt['content'], cur_response['content']))
	
			yield prompt_response

if __name__ == "__main__":
	sample_parser = WildcardParser()
	prompt_generator = sample_parser.get_prompt(False, True, False)
	
	for i in range(10):
		result = next(prompt_generator)

		print(result)
		print(len(result))
