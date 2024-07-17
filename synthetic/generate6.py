import argparse
import random
import json
import os
import time
import hashlib
import logging
import concurrent.futures
import requests
from string import Template
import csv
from nltk.corpus import wordnet
from textblob import TextBlob
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
from gensim.models import Word2Vec
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROMPT_TEMPLATES = [
    Template("Describe a complex project using $software that incorporates $feature and $technique."),
    Template("Write a step-by-step tutorial on achieving a $style effect in $software for a $project."),
    Template("Explain the role of $technique in developing a $project within $software, highlighting potential challenges."),
    Template("Create a detailed design brief for a $object using $software, focusing on $style and $feature elements.")
]

SOFTWARE = ["Photoshop", "Illustrator", "Premiere Pro", "After Effects", "Blender", "Figma", "GIMP", "Krita", "Inkscape", "DaVinci Resolve", "Final Cut Pro", "Unity", "Unreal Engine"]
FEATURES = ["photo manipulation", "vector illustration", "video editing", "3D modeling", "animation", "motion graphics", "visual effects", "color correction", "audio mixing", "user interface design", "game development", "virtual reality"]
STYLES = ["minimalist", "retro", "cyberpunk", "surreal", "glitch art", "low poly", "isometric", "watercolor", "line art", "typography-focused"]
TECHNIQUES = ["masking", "blending modes", "keyframing", "rigging", "texturing", "lighting", "compositing", "rotoscoping", "storyboarding", "user research"]
PROJECTS = ["short film", "music video", "animated series", "mobile game", "website design", "brand identity", "social media campaign", "virtual museum", "architectural visualization"]
OBJECTS = ["logo", "character model", "environment", "user interface", "3D asset", "video montage", "infographic", "poster", "book cover"]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate synthetic creative data with advanced features.")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of samples to generate in each batch")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of tokens for generated text")
    parser.add_argument("--interval", type=int, default=60, help="Interval between batch generations in seconds")
    parser.add_argument("--max_batches", type=int, default=None, help="Maximum number of batches to generate (None for infinite)")
    parser.add_argument("--complexity", choices=["low", "medium", "high"], default="medium", help="Complexity level of generated content")
    parser.add_argument("--output_format", choices=["json", "csv", "both"], default="both", help="Output format for generated data")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations of the data")
    return parser.parse_args()

def generate_prompt(prev_response=None, complexity="medium"):
    template = Template(f"Continue the creative discussion: {prev_response}\n") if prev_response else random.choice(PROMPT_TEMPLATES)
    substitutions = {
        "software": random.choice(SOFTWARE),
        "feature": random.choice(FEATURES),
        "style": random.choice(STYLES),
        "technique": random.choice(TECHNIQUES),
        "project": random.choice(PROJECTS),
        "object": random.choice(OBJECTS)
    }
    if complexity == "high":
        substitutions["additional_details"] = "Provide in-depth technical explanations, explore unconventional approaches, and discuss potential limitations or trade-offs."
    return template.substitute(**substitutions)

def generate_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def load_existing_data(filename="synthetic_creative_data.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        for item in data:
            if 'hash' not in item:
                item['hash'] = generate_hash(item['generated_text'])
        return data
    return []

def save_data(data, batch_count, format='json'):
    if format == 'json' or format == 'both':
        filename = f"synthetic_creative_data_batch_{batch_count}.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Data saved to {filename}")

    if format == 'csv' or format == 'both':
        csv_filename = f"synthetic_creative_data_batch_{batch_count}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["prompt", "generated_text", "hash"])
            writer.writeheader()
            writer.writerows(data)
        logging.info(f"Data saved to {csv_filename}")

def generate_text(prompt, max_tokens=200, model="http://localhost:1234/v1/completions"):
    url = model
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.7}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()  
        
        if 'generated_text' in response_data:
            return response_data['generated_text']
        elif 'choices' in response_data and isinstance(response_data['choices'], list):
            for choice in response_data['choices']:
                if 'text' in choice:
                    return choice['text'].strip() 

        logging.error(f"API response format unexpected: {response_data}")
        return None  
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error requesting or parsing response from LLM API: {e}")
        return None

def check_quality(text, prompt, existing_data):
    min_length = 50
    max_length = 500
    required_keywords = ["design", "create", "process", "steps"]
    if not text.strip():
        return False
    keyword_counts = Counter(text.lower().split())
    keyword_density = sum(keyword_counts.get(kw, 0) for kw in required_keywords) / len(text.split())
    density_threshold = 0.02
    if existing_data:
        vectorizer = TfidfVectorizer()
        existing_texts = [item['generated_text'] for item in existing_data]
        all_texts = existing_texts + [text]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        similarity_threshold = 0.8
        if max(cosine_similarities) >= similarity_threshold:
            return False
    return min_length <= len(text) <= max_length and keyword_density >= density_threshold

def replace_with_synonyms(text):
    for word, tag in TextBlob(text).tags:
        if tag.startswith('NN'):
            synonyms = wordnet.synsets(word)
            if synonyms:
                text = text.replace(word, synonyms[0].lemmas()[0].name())
    return text

def back_translate(text, target_lang='fr'):
    translator = Translator()
    intermediate = translator.translate(text, dest=target_lang).text
    return translator.translate(intermediate, dest='en').text

def contextual_augmentation(text, model):
    words = text.split()
    augmented_words = []
    for word in words:
        if word in model.wv.key_to_index:
            similar_words = model.wv.most_similar(word, topn=3)
            augmented_words.append(random.choice([word] + [w for w, _ in similar_words]))
        else:
            augmented_words.append(word)
    return ' '.join(augmented_words)

def advanced_text_augmentation(text, word2vec_model):
    augmentation_techniques = [
        lambda t: back_translate(t),
        lambda t: contextual_augmentation(t, word2vec_model),
        lambda t: replace_with_synonyms(t)
    ]
    return random.choice(augmentation_techniques)(text)

def generate_diverse_responses(prompt, max_tokens=200, num_beams=3):
    responses = []
    for _ in range(num_beams):
        response = generate_text(prompt, max_tokens)
        responses.append(response)
    return responses

def generate_word_cloud(texts):
    combined_text = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Generated Texts')
    plt.savefig('word_cloud.png')
    plt.close()

def visualize_topic_distribution(texts):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(doc_term_matrix)
    topic_results = lda.transform(doc_term_matrix)
    topic_names = [f'Topic {i+1}' for i in range(5)]
    df_topic_results = pd.DataFrame(topic_results, columns=topic_names)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_topic_results)
    plt.title('Distribution of Topics in Generated Texts')
    plt.ylabel('Topic Probability')
    plt.savefig('topic_distribution.png')
    plt.close()

def visualize_similarity_network(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='YlOrRd')
    plt.title('Similarity Network of Generated Texts')
    plt.savefig('similarity_network.png')
    plt.close()

def generate_creative_data(existing_data, num_samples=5, max_tokens=200, complexity="medium", word2vec_model=None):
    new_data = []
    existing_hashes = set(item.get('hash', generate_hash(item['generated_text'])) for item in existing_data)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_prompt = {}
        for _ in range(num_samples):
            prev_response = new_data[-1]["generated_text"] if new_data else None
            prompt = generate_prompt(prev_response, complexity)
            future_to_prompt[executor.submit(generate_diverse_responses, prompt, max_tokens)] = prompt

        for future in concurrent.futures.as_completed(future_to_prompt):
            prompt = future_to_prompt[future]
            try:
                responses = future.result()
                for response in responses:
                    response = advanced_text_augmentation(response, word2vec_model)
                    if check_quality(response, prompt, existing_data):
                        data_hash = generate_hash(response)
                        if data_hash not in existing_hashes:
                            new_data.append({
                                "prompt": prompt,
                                "generated_text": response,
                                "hash": data_hash
                            })
                            existing_hashes.add(data_hash)
                            if len(new_data) == num_samples:
                                return new_data
            except Exception as e:
                logging.error(f"Generation failed for prompt '{prompt}': {e}")
    return new_data

def continuous_data_generation(args):
    existing_data = load_existing_data()
    batch_count = 0
    word2vec_model = Word2Vec(sentences=[text.split() for text in [item['generated_text'] for item in existing_data]], vector_size=100, window=5, min_count=1, workers=4)

    while args.max_batches is None or batch_count < args.max_batches:
        new_data = generate_creative_data(existing_data, args.batch_size, args.max_tokens, args.complexity, word2vec_model)
        existing_data.extend(new_data)
        
        if args.output_format in ['json', 'both']:
            save_data(new_data, batch_count, 'json')
        if args.output_format in ['csv', 'both']:
            save_data(new_data, batch_count, 'csv')
        
        if args.visualize and batch_count % 5 == 0:  # Generate visualizations every 5 batches
            all_texts = [item['generated_text'] for item in existing_data]
            generate_word_cloud(all_texts)
            visualize_topic_distribution(all_texts)
            visualize_similarity_network(all_texts)
        
        batch_count += 1
        if args.max_batches is None or batch_count < args.max_batches:
            time.sleep(args.interval)

        # Update Word2Vec model with new data
        word2vec_model.build_vocab([item['generated_text'].split() for item in new_data], update=True)
        word2vec_model.train([item['generated_text'].split() for item in new_data], total_examples=len(new_data), epochs=1)

if __name__ == "__main__":
    args = parse_arguments()
    continuous_data_generation(args)