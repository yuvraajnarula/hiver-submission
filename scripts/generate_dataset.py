import os
import json
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()

def generate_dataset():
    prompt = """
    Generate a JSON array of 15 synthetic customer support emails for a SaaS product called 'Hiver' (a shared inbox tool inside Gmail). 
    Each object must have exactly these 4 keys:
    1. "id": integer
    2. "incoming_email": The customer's email. Vary the tone (angry, confused, polite) and topic (billing, login, feature request, bug).
    3. "internal_policy": A strict, 1-2 sentence internal company policy relevant to this specific email that the agent MUST follow.
    4. "golden_reply": The ideal, empathetic, and concise reply the agent sent, strictly adhering to the internal_policy.
    
    Output ONLY valid JSON. No markdown formatting, no explanations.
    """
    
    print("Generating dataset via Groq (Llama 3.3)...")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" },
        temperature=0.7
    )
    
    raw_content = response.choices[0].message.content
    
    # Fallback parsing just in case the model wraps it in markdown despite instructions
    if "```json" in raw_content:
        raw_content = re.search(r'```json(.*?)```', raw_content, re.DOTALL).group(1)
    elif "```" in raw_content:
        raw_content = re.search(r'```(.*?)```', raw_content, re.DOTALL).group(1)

    raw_json = json.loads(raw_content)
    
    # Handle different JSON structures the LLM might return
    if isinstance(raw_json, dict):
        dataset = raw_json.get("data", raw_json.get("emails", []))
        if not dataset: 
            # If it's just a dict with random keys, take the first list value
            for v in raw_json.values():
                if isinstance(v, list):
                    dataset = v
                    break
    else:
        dataset = raw_json

    with open("data/dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Successfully generated {len(dataset)} emails in data/dataset.json")

if __name__ == "__main__":
    generate_dataset()