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

def save_data(data, batch_count):
    filename = f"synthetic_creative_data_batch_{batch_count}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Data saved to {filename}")

    # Export to CSV
    csv_filename = f"synthetic_creative_data_batch_{batch_count}.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "generated_text", "hash"])
        writer.writeheader()
        writer.writerows(data)





def generate_text(prompt, max_tokens=200, model="http://localhost:1234/v1/completions"):
    url = model
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.7}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()  
        
        # Handle different possible response formats
        if 'generated_text' in response_data:
            return response_data['generated_text']
        elif 'choices' in response_data and isinstance(response_data['choices'], list):
            for choice in response_data['choices']:
                if 'text' in choice:
                    return choice['text'].strip() 

        # If none of the above, log the error and return None
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

def generate_diverse_responses(prompt, max_tokens=200, num_beams=3):
    responses = []
    for _ in range(num_beams):
        response = generate_text(prompt, max_tokens)
        responses.append(response)
    return responses

def generate_creative_data(existing_data, num_samples=5, max_tokens=200, complexity="medium"):
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
                    response = replace_with_synonyms(response)
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

def continuous_data_generation(batch_size=5, max_tokens=200, interval=60, max_batches=None, complexity="medium"):
    existing_data = load_existing_data()
    batch_count = 0
    while max_batches is None or batch_count < max_batches:
        new_data = generate_creative_data(existing_data, batch_size, max_tokens, complexity)
        existing_data.extend(new_data)
        save_data(new_data, batch_count)
        batch_count += 1
        if max_batches is None or batch_count < max_batches:
            time.sleep(interval)

if __name__ == "__main__":
    continuous_data_generation(batch_size=5, max_tokens=200, interval=60, max_batches=None, complexity="medium")
