import openai

def run_gpt(messages, model, max_tokens = 10, temperature = 0):
    assert model in ["gpt-4", "gpt-3.5-turbo", 'gpt-4-turbo', 'gpt-3.5-turbo-0613']
    
    if model == 'gpt-4-turbo':
        model = 'gpt-4-1106-preview'
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )

    data = response['choices'][0]['message']['content']
    parsed_data = [data.split(". ", 1)[1] for line in data.split("\n") if ". " in line]

    return (parsed_data)

def gen_failures(context, failure_mode, num_examples, model = 'gpt-3.5-turbo'):
    query = context + '\n' + "Failure Mode: [" + failure_mode + "]\n"
    messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': query}]

    failures = []

    for i in range(int(num_examples // 10)):
        llm_output = run_gpt(messages, model, max_tokens = 1000, temperature = 0)
        failures.extend(llm_output)

    return (failures)
    
news_context = """
Write down 10 example news articles that a language model would likely elicit a failure in understanding for the following failure mode. Please make each news article summary substantial in length, so that there are at least 150 words. To give you an idea of what a news article summary looks like, here are three examples of such formats: 
1. ''– Seems that choking a franchise ball player in the dugout will get you in trouble. Jonathan Papelbon, closer for the Washington Nationals, learned that Monday after attacking star outfielder Bryce Harper the day before in a dispute over Harper's playing style, the Washington Post reports. "I was upset," manager Matt Williams told the AP after seeing video of the altercation. "I was appalled." Indeed, Papelbon received a four-game suspension for the assault, and decided not to appeal a three-game Major League Baseball suspension for plunking a Baltimore Orioles player last week, which leaves Papelbon out for the season. The issue? Harper had flied out to left in the eighth inning and merely trotted over to first base. As Harper returned to the dugout, Papelbon angrily told him to "run it out," and Harper retorted with "let's [expletive] go." So the 34-year-old closer clutched the 22-year-old's neck and shoved him against the wall, until several Nats coaches and players pulled them apart, ESPN reports. For the record, Papelbon did apologize: "Yeah, he apologized. So, you know, whatever," says Harper, per the New York Times. "It’s like brothers fighting. That’s what happens." It's a sad note for a talented team that was widely expected to win the World Series when the season began. They were eliminated from contention on Saturday.'
2. '– For women who sleep long hours but wake up cursing the day, there's an app for that—or at least an app that corroborates your experience. The alarm-clock app Sleep Cycle gathered stats on one million users across 50 nations over nine months, and found that women aged 16 to 55 are getting more sleep than men, QZ.com reports. But in most countries (Ukraine, Portugal, and Colombia excepted) they self-report worse morning moods than do men. Why the grumpiness? Well, it echoes earlier Duke University research that women need more sleep, and will suffer mentally and physically without it, as Australia's News Network reported in 2013. "We found that women had more depression, women had more anger, and women had more hostility early in the morning," said sleep expert Michael Breus at the time. Breus' solution: Take naps of 25 or 90 minutes to rejuvenate the mind. "The more of your brain you use during the day, the more of it that needs to recover and, consequently, the more sleep you need," said another sleep expert. "Women tend to multi-task ... and so they use more of their actual brain than men do." A few years back, Arianna Huffington co-wrote a blog calling sleep "the next feminist issue" because overworked women are sleep-deprived. Indeed, the Duke research indicates that women's lack of sleep is more likely to cause serious health problems including depression, stroke, and heart disease. According to Sleep Cycle, Japanese women sleep the least (5 hours, 56 minutes) while Fins, the Dutch, and New Zealanders get the most (7 hours, 41 minutes), and American women average just over 7 hours. (Orange glasses could help you get better sleep.)'
3. '– The latest fallout over a Republican candidate's views on rape and/or abortion: A Seattle-area congressional nominee has dubbed abortions after rape "more violence onto a woman's body." Tea Party candidate John Koster was caught on tape noting that as to cases when the life of the mother is at stake, "I'm not going to make that decision," Reuters reports. "But on the rape thing, it's like, how does putting more violence onto a woman's body and taking the life of an innocent child that's a consequence of this crime, how does that make it better?" He added: "I know crime has consequences, but how does it make it better by killing a child?" the Seattle Times reports. Koster's Democratic opponent, Suzan DelBene, calls the comments "out of touch." Koster's website now says DelBene supporters were up to "dirty tricks" with the tape, released by a liberal activist group. "The recording was done secretly, then edited to suit DelBene's agenda," Koster's campaign manager says. "The insinuation that John Koster is in some way 'callous' or 'cavalier' when it comes to the subject of rape is another example of the vicious and desperate tactics ... employed to slander the good name of John Koster."'

You will be evaluated on how well you actually perform. Your sentence structure and
length can be creative; extrapolate based on the failure mode you’ve summarized. Be
both creative and cautious.
"""

cur_context = news_context
failure_modes_path = "failure_modes/tmp.txt"
failures = []

with open(failure_modes_path, 'r') as f:
    for line in f:
        line = line.strip()

        if line:
            print("Failure mode -> ", line)
            failures.extend(gen_failures(cur_context, line, 10))

output_file = "generation_output/news_indomain_all_failures.txt"

with open(output_file, 'w') as f:
    for fail in failures:
        f.write(f"{fail}\n")
        
