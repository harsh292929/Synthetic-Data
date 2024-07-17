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
from textblob import TextBlob
from nltk.corpus import wordnet

PROMPT_TEMPLATES = [
    Template("Describe a complex $software project involving $feature:"),
    Template("Write a tutorial on creating a $style in $software:"),
    Template("Explain the process of $technique a $project in $software:"),
    Template("Outline the steps to create a $object using $software:"),
]

SOFTWARE = ["Photoshop", "Illustrator", "Premiere Pro", "After Effects", "Blender", "Adobe XD", "Lightroom", "InDesign"]
FEATURES = ["multiple layers", "vector graphics", "3D modeling", "motion tracking", "color grading", "UI/UX design"]
STYLES = ["minimalist", "vintage", "futuristic", "abstract", "photorealistic"]
TECHNIQUES = ["compositing", "animating", "rendering", "retouching"]
PROJECTS = ["short film", "advertisement", "social media campaign", "website mockup"]
OBJECTS = ["character", "landscape", "product", "infographic"]

def generate_prompt():
    template = random.choice(PROMPT_TEMPLATES)
    return template.substitute(
        software=random.choice(SOFTWARE),
        feature=random.choice(FEATURES),
        style=random.choice(STYLES),
        technique=random.choice(TECHNIQUES),
        project=random.choice(PROJECTS),
        object=random.choice(OBJECTS)
    )

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

def generate_text(prompt, max_tokens=200):
    url = "http://localhost:1234/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['text']
    except requests.RequestException as e:
        return None

def check_quality(text):
    min_length = 50
    max_length = 500
    required_keywords = ["design", "create", "process", "steps"]
    return min_length <= len(text) <= max_length and any(keyword in text.lower() for keyword in required_keywords)

def replace_with_synonyms(text):
    for word, tag in TextBlob(text).tags:
        if tag.startswith('NN'):
            synonyms = wordnet.synsets(word)
            if synonyms:
                text = text.replace(word, synonyms[0].lemmas()[0].name())
    return text


def generate_creative_data(existing_data, num_samples=5, max_tokens=200):
    new_data = []
    existing_hashes = set(item.get('hash', generate_hash(item['generated_text'])) for item in existing_data)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_prompt = {executor.submit(generate_text, generate_prompt(), max_tokens): generate_prompt() for _ in range(num_samples * 2)}
        for future in concurrent.futures.as_completed(future_to_prompt):
            prompt = future_to_prompt[future]
            try:
                result = future.result()
                result = replace_with_synonyms(result)  # Apply synonym replacement
                if result and check_quality(result):
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
                pass  # Handle exceptions silently in this simplified version

    return new_data

def continuous_data_generation(batch_size=5, max_tokens=200, interval=60, max_batches=None):
    filename = "synthetic_creative_data1.json"
    existing_data = load_existing_data(filename)
    batch_count = 0

    while max_batches is None or batch_count < max_batches:
        new_data = generate_creative_data(existing_data, batch_size, max_tokens)
        existing_data.extend(new_data)
        save_data_to_file(existing_data, filename)
        export_to_csv(existing_data)

        batch_count += 1

        if max_batches is None or batch_count < max_batches:
            time.sleep(interval)

if __name__ == "__main__":
    continuous_data_generation(batch_size=5, max_tokens=200, interval=60, max_batches=None)
