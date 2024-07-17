import requests
import json

def generate_text(prompt, max_tokens=100):
    url = "http://localhost:1234/v1/completions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()['choices'][0]['text']
    else:
        return f"Error: {response.status_code}, {response.text}"

# Test the connection
test_prompt = "Hello, LLama!"
result = generate_text(test_prompt)
print(result)