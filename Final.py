from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


user_input = input("Enter topic and category separated by a comma: ")

topic, category = [x.strip() for x in user_input.split(',')]
print("Topic is", topic)
print("Category is", category)
client = OpenAI(
  api_key="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"
)

#generate search query using ChatGPT
response = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "Generate a concise and effective Google search query for use with the Custom Search API to retrieve the 10 most relevant" 
     " and informative sources on the topic: '" + topic + "' in the category: '" + category + "'. Do not include specific website names or URLs. Output only"
      " the search query string and nothing else."}
  ]
)
query = response.choices[0].message.content
print(query)

# make the search and get urls and titles
params = {
    "key": "AIzaSyDZIRv_gba5v_kCBOIxue6HziZJtElA5ho",
    "cx": "c56384b71d7394782",
    "q": query,
    "num": 10
}
response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
results = response.json()
sources = []
for result in results.get("items", []):
    sources.append({
        'title': result.get("title"),
        'url': result.get("link")
    })

def extract_text(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/77.0.3865.90 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, timeout=(5, 15))
        response.raise_for_status() 

        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        lines = []
        for p in paragraphs:
            lines.append(p.get_text(strip=True))
        return "\n".join(lines)

    except Exception as e:
        return f"Failed to extract text: {e}"

#extract text from urls and remove irrelevant information  
filtered_sources = []
for source in sources:
    print(f"Extracting text and removing irrelevant information from: {source['title'], source['url']}")
    text = extract_text(source['url'])
    if not text:
        continue
    filter_response = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
             {"role": "user", "content": "Remove irrelevant text from the following text. Only keep information that is directly related to the topic '" +
              topic + "' and category '" + category + "'. Output only relevant content if there is relevant content and only and an empty message if " 
              "the whole source is irrelevant: \n\n" + text}
        ]
    ) 
    filtered_source = filter_response.choices[0].message.content
    if filtered_source.strip():  
        filtered_sources.append({
            'text': filtered_source,
            'title': source['title'],
            'url': source['url']
        })

#deduplicate filtered sources, keeping longer source if duplicate found
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = []
for source in filtered_sources:
    texts.append(source['text'])
embeddings = []
for text in texts:
    emb = model.encode(text, convert_to_numpy=True)
    embeddings.append(emb)

embeddings = np.vstack(embeddings)
cosine_scores = cosine_similarity(embeddings)
threshold = 0.825
deduplicated_sources = []
used = set()
index_map = {}

b=True
for i in range(len(filtered_sources)):
    if i in used:
        continue
    longest=i
    for j in range(i + 1, len(filtered_sources)):
        if cosine_scores[i][j] >= threshold:
            print(f"Removing duplicate of:\n'{filtered_sources[i]['title']}' and '{filtered_sources[j]['title']}'")
            b=False
            used.add(j)
            if len(filtered_sources[j]['text']) > len(filtered_sources[longest]['text']):
                longest = j
    deduplicated_sources.append(filtered_sources[longest])
    used.add(longest)
if(b): print("No duplicates found")