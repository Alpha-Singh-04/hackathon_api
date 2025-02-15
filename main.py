from utils import extract_article_content, split_text_into_chunks

URL = "https://www.hindustantimes.com/india-news/second-flight-of-119-indian-illegal-immigrants-from-us-arrive-in-amritsar-today-february-15-101739582916039.html"

title, content = extract_article_content(URL)


# abhi ye sub chunks mei banana hai !!

chunks = split_text_into_chunks(content)



# now model part

from transformers import pipeline

# Initialize the text classification pipeline
clf = pipeline("text-classification", model="jy46604790/Fake-News-Bert-Detect")

# Analyze each chunk
results = []
for chunk in chunks:
    result = clf(chunk)
    results.append(result)



results = [
    [{'label': 'LABEL_0', 'score': 0.9972700476646423}],
    [{'label': 'LABEL_0', 'score': 0.9995183944702148}],
    [{'label': 'LABEL_0', 'score': 0.9992775321006775}],
    [{'label': 'LABEL_1', 'score': 0.9999017715454102}]
]

flattened_results = [item for sublist in results for item in sublist]

for result in flattened_results:
    if result['label'] == 'LABEL_1':
        overall_result = (True, result['score'])
        break
else:
    
    highest_fake_score = max(result['score'] for result in flattened_results if result['label'] == 'LABEL_0')
    overall_result = (False, highest_fake_score)

print(f"Overall result: {overall_result}")


  

