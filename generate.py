import random
import json
import os
from connect import generate_text
import time
import hashlib

# import random
# import json
# import os
# import time
# import hashlib

prompts = [
    "Describe a complex Photoshop project involving multiple layers and effects:",
    "Write a tutorial on creating a vector illustration in Adobe Illustrator:",
    "Explain the process of color grading a short film in Adobe Premiere Pro:",
    "Outline the steps to create a 3D model of a character in Blender:",
    "Describe an innovative UI/UX design created using Adobe XD:",
    "Detail the workflow for editing a portrait photo in Lightroom:",
    "Explain how to create a motion graphics intro using After Effects:",
    "Describe a challenging logo design process using Illustrator and Photoshop:",
    "Write about integrating AR elements into a marketing campaign using Adobe Aero:",
    "Outline the process of creating an e-book layout in InDesign:"
]







def generate_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def load_existing_data(filename="synthetic_creative_data.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []

def save_data_to_file(data, filename="synthetic_creative_data.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {filename}")

def generate_creative_data(existing_data, num_samples=5, max_tokens=200):
    new_data = []
    existing_hashes = set(item['hash'] for item in existing_data)
    
    while len(new_data) < num_samples:
        prompt = random.choice(prompts)
        result = generate_text(prompt, max_tokens)
        data_hash = generate_hash(result)
        
        if data_hash not in existing_hashes:
            new_data.append({
                "prompt": prompt,
                "generated_text": result,
                "hash": data_hash
            })
            existing_hashes.add(data_hash)
    
    return new_data

def continuous_data_generation(batch_size=5, max_tokens=200, interval=60, max_batches=None):
    filename = "synthetic_creative_data.json"
    existing_data = load_existing_data(filename)
    batch_count = 0
    
    while max_batches is None or batch_count < max_batches:
        print(f"Generating batch {batch_count + 1}")
        new_data = generate_creative_data(existing_data, batch_size, max_tokens)
        existing_data.extend(new_data)
        save_data_to_file(existing_data, filename)
        
        for item in new_data:
            print(f"Prompt: {item['prompt']}\n")
            print(f"Generated Text: {item['generated_text']}\n")
            print("-" * 50)
        
        batch_count += 1
        
        if max_batches is None or batch_count < max_batches:
            print(f"Waiting {interval} seconds before generating next batch...")
            time.sleep(interval)

# Start continuous data generation
continuous_data_generation(batch_size=5, max_tokens=200, interval=60, max_batches=None)