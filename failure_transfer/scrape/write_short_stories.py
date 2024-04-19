from scrape_stories import StoriesScrape

total_examples = 1000
stories_scraper = StoriesScrape()
original = stories_scraper.get_examples(total_examples)

with open("./data/example_short_stories.txt", "w") as f:
    for story in original:
        f.write(f"{story}\n")

    