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
import re
import uuid
from datetime import datetime
from collections import defaultdict, Counter
from nltk.corpus import wordnet
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tenacity import retry, stop_after_attempt, wait_exponential

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

class SimpleWordEmbedding:
    def __init__(self, texts, vector_size=100):
        self.word_to_id = {}
        self.id_to_word = {}
        self.embeddings = []
        self.vector_size = vector_size
        self.build_vocab(texts)

    def build_vocab(self, texts, update=False):
        words = set()
        for text in texts:
            words.update(text.split())
        
        if not update:
            self.word_to_id = {word: i for i, word in enumerate(words)}
            self.id_to_word = {i: word for word, i in self.word_to_id.items()}
            self.embeddings = [
                [random.uniform(-1, 1) for _ in range(self.vector_size)]
                for _ in range(len(self.word_to_id))
            ]
        else:
            current_size = len(self.word_to_id)
            for word in words:
                if word not in self.word_to_id:
                    self.word_to_id[word] = current_size
                    self.id_to_word[current_size] = word
                    self.embeddings.append([random.uniform(-1, 1) for _ in range(self.vector_size)])
                    current_size += 1

    def most_similar(self, word, topn=3):
        if word not in self.word_to_id:
            return []
        word_id = self.word_to_id[word]
        word_embedding = self.embeddings[word_id]
        
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            if i != word_id:
                similarity = sum(a * b for a, b in zip(word_embedding, embedding))
                similarities.append((self.id_to_word[i], similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:topn]

class DatasetVersion:
    def __init__(self, data, parent=None):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.now()
        self.data = data
        self.parent = parent
        self.children = []
        self.ratings = defaultdict(list)  # User ratings for each data item
        self.tags = defaultdict(set)  # User tags for each data item

    def add_child(self, child):
        self.children.append(child)

class VersionControlSystem:
    def __init__(self):
        self.versions = {}
        self.current_version = None

    def create_version(self, data, parent=None):
        new_version = DatasetVersion(data, parent)
        self.versions[new_version.id] = new_version
        if parent:
            parent.add_child(new_version)
        self.current_version = new_version
        return new_version

    def get_version(self, version_id):
        return self.versions.get(version_id)

    def revert_to_version(self, version_id):
        if version_id in self.versions:
            self.current_version = self.versions[version_id]
            return self.current_version
        return None

    def merge_versions(self, version_id1, version_id2):
        v1 = self.versions.get(version_id1)
        v2 = self.versions.get(version_id2)
        if v1 and v2:
            merged_data = v1.data + v2.data  # Simple concatenation, you might want a more sophisticated merge
            return self.create_version(merged_data, parent=v1)
        return None

    def add_rating(self, version_id, data_index, user_id, rating):
        version = self.versions.get(version_id)
        if version:
            version.ratings[data_index].append((user_id, rating))

    def add_tag(self, version_id, data_index, user_id, tag):
        version = self.versions.get(version_id)
        if version:
            version.tags[data_index].add((user_id, tag))

    def get_average_ratings(self, version_id):
        version = self.versions.get(version_id)
        if version:
            return {index: sum(r for _, r in ratings) / len(ratings) 
                    for index, ratings in version.ratings.items()}
        return {}

    def get_popular_tags(self, version_id, top_n=5):
        version = self.versions.get(version_id)
        if version:
            tag_counts = defaultdict(int)
            for tags in version.tags.values():
                for _, tag in tags:
                    tag_counts[tag] += 1
            return sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return []

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
    if not data:
        logging.warning(f"No data to save for batch {batch_count}")
        return

    if format == 'json' or format == 'both':
        filename = f"synthetic_creative_data_batch_{batch_count}.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Data saved to {filename}")

    if format == 'csv' or format == 'both':
        csv_filename = f"synthetic_creative_data_batch_{batch_count}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            if data and isinstance(data[0], dict):
                fieldnames = data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
                logging.info(f"Data saved to {csv_filename}")
            else:
                logging.error(f"Invalid data format for CSV writing: {data}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_text(prompt, max_tokens=200, model="http://localhost:1234/v1/completions"):
    url = model
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.7}
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        response_data = response.json()  
        
        generated_text = None
        if 'generated_text' in response_data:
            generated_text = response_data['generated_text']
        elif 'choices' in response_data and isinstance(response_data['choices'], list):
            for choice in response_data['choices']:
                if 'text' in choice:
                    generated_text = choice['text'].strip()
                    break

        if not generated_text:
            raise ValueError("Empty response from API")

        return generated_text
        
    except (requests.exceptions.RequestException, ValueError) as e:
        logging.error(f"Error requesting or parsing response from LLM API: {e}")
        raise

def check_quality(text, prompt, existing_data):
    if not text or not text.strip():
        return False
    
    min_length = 20
    max_length = 1000
    required_keywords = ["design", "create", "process", "steps", "project", "software"]
    
    keyword_counts = Counter(text.lower().split())
    keyword_density = sum(keyword_counts.get(kw, 0) for kw in required_keywords) / len(text.split())
    density_threshold = 0.01
    
    if existing_data:
        vectorizer = TfidfVectorizer()
        existing_texts = [item['generated_text'] for item in existing_data]
        all_texts = existing_texts + [text]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        similarity_threshold = 0.9
        if max(cosine_similarities) >= similarity_threshold:
            return False
    
    return min_length <= len(text) <= max_length and keyword_density >= density_threshold

def replace_with_synonyms(text):
    try:
        for word, tag in TextBlob(text).tags:
            if tag.startswith('NN'):
                synonyms = wordnet.synsets(word)
                if synonyms:
                    text = text.replace(word, synonyms[0].lemmas()[0].name())
        return text
    except Exception as e:
        logging.error(f"Error in replace_with_synonyms: {e}")
        return text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def back_translate(text, target_lang='fr'):
    try:
        translator = Translator()
        if not text:
            logging.warning("Received empty or None text for back translation")
            return text

        time.sleep(1)
        
        intermediate = translator.translate(text, dest=target_lang)
        if not intermediate or not intermediate.text:
            logging.warning(f"Failed to translate to {target_lang}")
            return text
        
        time.sleep(1)
        
        result = translator.translate(intermediate.text, dest='en')
        if not result or not result.text:
            logging.warning("Failed to translate back to English")
            return text
        
        return result.text
    except Exception as e:
        logging.error(f"Error in back_translate: {e}")
        return text

def contextual_augmentation(text, model):
    words = text.split()
    augmented_words = []
    for word in words:
        if word in model.word_to_id:
            similar_words = model.most_similar(word, topn=3)
            augmented_words.append(random.choice([word] + [w for w, _ in similar_words]))
        else:
            augmented_words.append(word)
    return ' '.join(augmented_words)

def advanced_text_augmentation(text, word_embedding_model):
    if not text:
        logging.error("Received empty text for augmentation")
        return text
    
    augmentation_techniques = [
        lambda t: replace_with_synonyms(t),
        lambda t: contextual_augmentation(t, word_embedding_model)
    ]
    try:
        return random.choice(augmentation_techniques)(text)
    except Exception as e:
        logging.error(f"Error during text augmentation: {e}")
        return text 

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

def improve_generation_based_on_feedback(vcs, version_id, generate_function):
    version = vcs.get_version(version_id)
    if not version:
        return None

    avg_ratings = vcs.get_average_ratings(version_id)
    popular_tags = dict(vcs.get_popular_tags(version_id))

    # Filter out highly-rated items
    good_items = [item for i, item in enumerate(version.data) if avg_ratings.get(i, 0) >= 4.0]

    # Use good items and popular tags to guide new generation
    new_data = generate_function(existing_data=good_items, 
                                 num_samples=len(version.data), 
                                 preferred_tags=popular_tags)

    return vcs.create_version(new_data, parent=version)

def generate_creative_data(existing_data, num_samples=5, max_tokens=200, complexity="medium", word_embedding_model=None, preferred_tags=None):
    new_data = []
    existing_hashes = set(item.get('hash', generate_hash(item['generated_text'])) for item in existing_data)
    max_attempts = num_samples * 3

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_prompt = {}
        for _ in range(max_attempts):
            prev_response = new_data[-1]["generated_text"] if new_data else None
            prompt = generate_prompt(prev_response, complexity)
            if preferred_tags:
                prompt += f" Consider incorporating these themes: {', '.join(preferred_tags.keys())}"
            future_to_prompt[executor.submit(generate_diverse_responses, prompt, max_tokens)] = prompt

        for future in concurrent.futures.as_completed(future_to_prompt):
            prompt = future_to_prompt[future]
            try:
                responses = future.result()
                logging.info(f"Generated {len(responses)} responses for prompt: {prompt[:50]}...")
                for response in responses:
                    if not response:
                        logging.warning(f"Received empty response for prompt: {prompt[:50]}...")
                        continue
                    
                    response = advanced_text_augmentation(response, word_embedding_model)
                    if check_quality(response, prompt, existing_data):
                        data_hash = generate_hash(response)
                        if data_hash not in existing_hashes:
                            new_data.append({
                                "prompt": prompt,
                                "generated_text": response,
                                "hash": data_hash
                            })
                            existing_hashes.add(data_hash)
                            logging.info(f"Added new data item. Total: {len(new_data)}")
                            if len(new_data) == num_samples:
                                return new_data
            except Exception as e:
                logging.error(f"Generation failed for prompt '{prompt[:50]}...': {e}")
                logging.exception("Full traceback:")

    logging.info(f"Generated {len(new_data)} new data items out of {num_samples} requested")
    return new_data

def continuous_data_generation(args):
    existing_data = load_existing_data()
    batch_count = 0
    word_embedding_model = SimpleWordEmbedding([item['generated_text'] for item in existing_data])
    vcs = VersionControlSystem()

    while args.max_batches is None or batch_count < args.max_batches:
        new_data = generate_creative_data(existing_data, args.batch_size, args.max_tokens, args.complexity, word_embedding_model)
        
        if not new_data:
            logging.warning(f"No new data generated in batch {batch_count}")
            time.sleep(args.interval)
            continue

        # Create a new version
        new_version = vcs.create_version(new_data)
        existing_data.extend(new_data)
        
        if args.output_format in ['json', 'both']:
            save_data(new_data, batch_count, 'json')
        if args.output_format in ['csv', 'both']:
            save_data(new_data, batch_count, 'csv')
        
        # Simulate user feedback (in a real scenario, this would come from user interaction)
        for i, item in enumerate(new_data):
            vcs.add_rating(new_version.id, i, f"user{i}", random.randint(1, 5))
            vcs.add_tag(new_version.id, i, f"user{i}", random.choice(["creative", "innovative", "needs_improvement"]))

        # Every 5 batches, generate an improved version based on feedback
        if batch_count % 5 == 0:
            improved_version = improve_generation_based_on_feedback(vcs, new_version.id, generate_creative_data)
            if improved_version:
                logging.info(f"Created improved version {improved_version.id} based on feedback")

        if args.visualize and batch_count % 5 == 0:
            all_texts = [item['generated_text'] for item in existing_data]
            generate_word_cloud(all_texts)
            visualize_topic_distribution(all_texts)
            visualize_similarity_network(all_texts)
        
        batch_count += 1
        if args.max_batches is None or batch_count < args.max_batches:
            time.sleep(args.interval)

        # Update word embedding model with new data
        word_embedding_model.build_vocab([item['generated_text'] for item in new_data], update=True)

if __name__ == "__main__":
    args = parse_arguments()
    
    # Test LLM API connection
    test_prompt = "Hello, world!"
    test_response = generate_text(test_prompt, max_tokens=10)
    if test_response is None:
        logging.error("Failed to connect to LLM API. Please check your connection and try again.")
        exit(1)
    else:
        logging.info("Successfully connected to LLM API.")
    
    continuous_data_generation(args)