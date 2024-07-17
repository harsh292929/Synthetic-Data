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

YOUR_API_KEY = '8162698651984c6da28c0e086aaae6a9'
YOUR_SEARCH_ENGINE_ID = 'p8e-FhNT_tQC2umcaUiJjxdAWBulcFT_Sin4'
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

def generate_prompt(prev_response=None, complexity="medium"):
    if prev_response:
        template = Template(f"Continue the creative discussion: {prev_response}\n")
    else:
        template = random.choice(PROMPT_TEMPLATES)
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

def save_data_to_file(data, filename="synthetic_creative_data.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def export_to_csv(data, filename="synthetic_creative_data.csv"):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "generated_text", "hash"])
        writer.writeheader()
        writer.writerows(data)

def generate_text(prompt, max_tokens=200, model="http://localhost:1234/v1/completions"): 
    url = model
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt, "max_tokens": max_tokens}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['generated_text']

def check_quality(text, prompt, existing_data):
    min_length = 50
    max_length = 500
    required_keywords = ["design", "create", "process", "steps"]
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

def verify_factuality(text, prompt):
    # Sample implementation using Google Search API (replace with your actual API):
    try:
        search_url = f"https://www.googleapis.com/customsearch/v1?key=YOUR_API_KEY&cx=YOUR_SEARCH_ENGINE_ID&q={prompt}"
        response = requests.get(search_url)
        response.raise_for_status()
        search_results = response.json()
        for result in search_results['items']:
            if text in result['snippet']:
                return True  # Found similar text in search results
    except requests.RequestException as e:
        logging.error(f"Fact-checking failed: {e}")
    return False  # Not found in search results

def generate_creative_data(existing_data, num_samples=5, max_tokens=200, complexity="medium"):
    new_data = []
    existing_hashes = set(item.get('hash', generate_hash(item['generated_text'])) for item in existing_data)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_prompt = {}
        for _ in range(num_samples * 2): 
            prev_response = new_data[-1]["generated_text"] if new_data else None
            prompt = generate_prompt(prev_response, complexity)
            future_to_prompt[executor.submit(generate_text, prompt, max_tokens)] = prompt

        for future in concurrent.futures.as_completed(future_to_prompt):
            prompt = future_to_prompt[future]
            try:
                result = future.result()
                result = replace_with_synonyms(result)
                if result and check_quality(result, prompt, existing_data) and verify_factuality(result, prompt):
                    data_hash = generate_hash(result)
                    if data_hash not in existing_hashes:
                        new_data.append({
                            "prompt": prompt,
                            "generated_text": result,
                            "hash": data_hash
                        })
                        existing_hashes.add(data_hash)
                    if len(new_data) == num_samples:
                        break
            except Exception as e:
                logging.error(f"Generation failed for prompt '{prompt}': {e}")
    return new_data

def continuous_data_generation(batch_size=5, max_tokens=200, interval=60, max_batches=None, complexity="medium"):
    filename = "synthetic_creative_data.json"
    existing_data = load_existing_data(filename)
    batch_count = 0
    while max_batches is None or batch_count < max_batches:
        new_data = generate_creative_data(existing_data, batch_size, max_tokens, complexity)
        existing_data.extend(new_data)
        save_data_to_file(existing_data, filename)
        export_to_csv(existing_data)  
        batch_count += 1
        if max_batches is None or batch_count < max_batches:
            time.sleep(interval)

if __name__ == "__main__":
    continuous_data_generation(batch_size=5, max_tokens=200, interval=60, max_batches=None, complexity="medium")
