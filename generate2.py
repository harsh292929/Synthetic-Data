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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from flask import Flask, jsonify, request
import threading
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurable prompt templates
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
    logging.info(f"Data saved to {filename}")

def export_to_csv(data, filename="synthetic_creative_data.csv"):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "generated_text", "hash"])
        writer.writeheader()
        writer.writerows(data)
    logging.info(f"Data exported to CSV: {filename}")


class DataGenerationSystem:
    def __init__(self):
        self.data = load_existing_data()
        self.vectorizer = TfidfVectorizer()
        self.prompt_performance = Counter()
        self.sentiment_distribution = {'positive': 0, 'neutral': 0, 'negative': 0}

    def generate_text(self, prompt, max_tokens=200):
        # Replace this URL with your local LLama 3 API endpoint
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
            logging.error(f"API request failed: {e}")
            return None

    def check_quality(self, text):
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Update sentiment distribution
        if sentiment > 0.1:
            self.sentiment_distribution['positive'] += 1
        elif sentiment < -0.1:
            self.sentiment_distribution['negative'] += 1
        else:
            self.sentiment_distribution['neutral'] += 1

        # Quality criteria
        min_length = 50
        max_length = 500
        required_keywords = ["design", "create", "process", "steps"]
        
        keyword_check = any(keyword in text.lower() for keyword in required_keywords)
        length_check = min_length <= len(text) <= max_length
        sentiment_check = -0.5 <= sentiment <= 0.8
        subjectivity_check = 0.2 <= subjectivity <= 0.8

        return all([keyword_check, length_check, sentiment_check, subjectivity_check])

    def semantic_similarity_check(self, new_text):
        if not self.data:
            return True
        existing_texts = [item['generated_text'] for item in self.data]
        all_texts = existing_texts + [new_text]
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        return max(cosine_similarities) < 0.8

    def generate_creative_data(self, num_samples=5, max_tokens=200):
        new_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_prompt = {executor.submit(self.generate_text, generate_prompt(), max_tokens): generate_prompt() for _ in range(num_samples * 2)}
            for future in concurrent.futures.as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    if result and self.check_quality(result) and self.semantic_similarity_check(result):
                        data_hash = generate_hash(result)
                        new_data.append({
                            "prompt": prompt,
                            "generated_text": result,
                            "hash": data_hash,
                            "tags": self.generate_tags(result),
                            "sentiment": TextBlob(result).sentiment.polarity
                        })
                        self.prompt_performance[prompt] += 1
                        if len(new_data) == num_samples:
                            break
                except Exception as e:
                    logging.error(f"Generation failed for prompt '{prompt}': {e}")
        
        self.data.extend(new_data)
        return new_data

    def generate_tags(self, text):
        blob = TextBlob(text)
        return [word for word, pos in blob.tags if pos.startswith('NN') or pos.startswith('JJ')][:5]

    def optimize_prompts(self):
        best_prompts = self.prompt_performance.most_common(5)
        return [prompt for prompt, _ in best_prompts]

    def visualize_data(self):
        plt.figure(figsize=(15, 10))

        # Sentiment distribution
        plt.subplot(2, 2, 1)
        sns.barplot(x=list(self.sentiment_distribution.keys()), y=list(self.sentiment_distribution.values()))
        plt.title('Sentiment Distribution')

        # Top tags
        all_tags = [tag for item in self.data for tag in item['tags']]
        top_tags = Counter(all_tags).most_common(10)
        plt.subplot(2, 2, 2)
        sns.barplot(x=[tag[0] for tag in top_tags], y=[tag[1] for tag in top_tags])
        plt.title('Top 10 Tags')
        plt.xticks(rotation=45)

        # Prompt performance
        top_prompts = self.prompt_performance.most_common(5)
        plt.subplot(2, 2, 3)
        sns.barplot(x=[prompt[:20] + '...' for prompt, _ in top_prompts], y=[count for _, count in top_prompts])
        plt.title('Top 5 Performing Prompts')
        plt.xticks(rotation=45)

        # Data growth over time
        plt.subplot(2, 2, 4)
        plt.plot(range(len(self.data)), [i+1 for i in range(len(self.data))])
        plt.title('Cumulative Data Growth')
        plt.xlabel('Generations')
        plt.ylabel('Total Data Points')

        plt.tight_layout()
        plt.savefig('data_visualization.png')
        logging.info("Data visualization saved as 'data_visualization.png'")

    def run(self, batch_size=5, max_tokens=200, interval=60, max_batches=None):
        batch_count = 0
        try:
            while max_batches is None or batch_count < max_batches:
                logging.info(f"Generating batch {batch_count + 1}")
                new_data = self.generate_creative_data(batch_size, max_tokens)
                save_data_to_file(self.data)
                export_to_csv(self.data)

                if batch_count % 10 == 0:
                    self.visualize_data()

                batch_count += 1

                if max_batches is None or batch_count < max_batches:
                    logging.info(f"Waiting {interval} seconds before generating next batch...")
                    time.sleep(interval)
        except KeyboardInterrupt:
            logging.info("Data generation interrupted by user.")
        finally:
            logging.info(f"Total batches generated: {batch_count}")
            logging.info(f"Total data points: {len(self.data)}")

# Flask routes for API
@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        'total_data_points': len(system.data),
        'sentiment_distribution': system.sentiment_distribution,
        'top_prompts': system.prompt_performance.most_common(5)
    })

@app.route('/generate', methods=['POST'])
def trigger_generation():
    batch_size = request.json.get('batch_size', 5)
    max_tokens = request.json.get('max_tokens', 200)
    new_data = system.generate_creative_data(batch_size, max_tokens)
    return jsonify({'new_data_points': len(new_data)})
@app.route('/')  # Add this route
def index():
    return "Welcome to the Creative Data Generator!"  # Or return a rendered templat
if __name__ == "__main__":
    system = DataGenerationSystem()
    
    # Start the data generation in a separate thread
    generation_thread = threading.Thread(target=system.run)
    generation_thread.start()

    # Start the Flask app
    app.run(debug=True, use_reloader=False)