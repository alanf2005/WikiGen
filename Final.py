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
    {"role": "user", "content": "Generate a Google search query for use with the Custom Search API to retrieve the 10 most relevant" 
     " and informative sources that can be used to make a wikipedia article on the topic: '" + topic + "' in the category: '" + category + "'. Do not include specific website names or URLs. Output only"
      " the search query string and nothing else."}
  ]
)
query = response.choices[0].message.content
print(query)

# make the search using Custom Search API, getting urls and titles
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
             {"role": "user", "content": "Remove irrelevant text from the following text. Only keep information that is directly related to information about the topic '" +
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

#Use sentence transformer on each source to get vectors
#Vertically stack vectors into a matrix(each row is a source) 
#Get the cosine similarity matrix, then compare each rowkeeping longer source if duplicate found
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = []
for source in filtered_sources:
    texts.append(source['text'])
embedding_vectors = []
for text in texts:
    emb = model.encode(text, convert_to_numpy=True)
    embedding_vectors.append(emb)

embeddings_matrix = np.vstack(embedding_vectors) 
cosine_scores = cosine_similarity(embeddings_matrix) #cosine_scores[i][j] means similarity between rows i and j in embeddings
threshold = 0.85
deduplicated_sources = []
used = set()
b=True
for i in range(len(filtered_sources)):
    if i in used:
        continue
    longest_index = i

    duplicates = [i]
    for j in range(i + 1, len(filtered_sources)):
        if j not in used and cosine_scores[i][j] >= threshold:
            duplicates.append(j)
            used.add(j)

            if len(filtered_sources[j]['text']) > len(filtered_sources[longest_index]['text']):
                longest_index = j

    used.add(i)  # Add i itself to the used set
    deduplicated_sources.append(filtered_sources[longest_index])
    used.add(i)
if(b): print("No duplicates found")

n=1
for source in deduplicated_sources:
    print("SOURCE " + str(n) + ": " + source['title'], "LINK:" + source['url'] + "\n", source['text'])
    n+=1