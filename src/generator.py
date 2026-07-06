import os
import sys
import json
from dotenv import load_dotenv
from groq import Groq

# Add src to path so we can import the retriever easily
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.retriever import InMemoryRetriever

load_dotenv()
client = Groq()

def format_context(retrieved_items):
    """Formats the retrieved RAG items into a readable string for the prompt."""
    context_str = ""
    for i, item in enumerate(retrieved_items, 1):
        context_str += f"""
--- Example {i} ---
Customer Email: {item['incoming_email']}
Internal Policy Applied: {item['internal_policy']}
Agent Reply: {item['golden_reply']}
"""
    return context_str

def generate_reply(new_email, retriever):
    """Generates a suggested reply for a new email using Groq and RAG context."""
    
    # 1. Retrieve grounded context
    print(f"Retrieving context for: '{new_email[:40]}...'")
    retrieved_items = retriever.search(new_email, top_k=2)
    context_str = format_context(retrieved_items)
    
    # 2. Construct the Prompt
    system_prompt = """You are an expert customer support agent for Hiver, a shared inbox tool for Gmail. 
Your goal is to write empathetic, concise, and highly accurate replies. 
You must strictly adhere to the internal policies provided in the examples. 
Do not hallucinate features or policies. If the policy says no refund, do not offer a refund."""

    user_prompt = f"""Here are examples of past customer emails, the strict internal policies applied to them, and the ideal agent replies. Use these to understand the tone and the rules you must follow.

{context_str}

--- NEW INCOMING EMAIL ---
{new_email}

--- INSTRUCTIONS ---
Draft a suggested reply for the NEW INCOMING EMAIL. 
- Be empathetic and professional.
- Be concise (customers want quick answers).
- Strictly follow the policies demonstrated in the examples above.
- Output ONLY the email reply text. Do not include "Subject:", "Dear Customer", or any meta-commentary."""

    # 3. Call Groq (Llama 3.3 70B)
    print("Generating reply via Groq...")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3, # Low temperature for strict policy adherence
        max_tokens=300
    )
    
    generated_reply = response.choices[0].message.content.strip()
    
    return {
        "incoming_email": new_email,
        "retrieved_context": retrieved_items,
        "generated_reply": generated_reply
    }

def main():
    # Initialize the retriever (loads dataset & embeddings)
    retriever = InMemoryRetriever()
    
    # Define a few test emails to generate replies for
    test_emails = [
        "Hi, I was double charged for my Pro plan this month. Can you fix this immediately?",
        "I want to cancel my subscription. How do I do that? Also, your UI is very confusing.",
        "Is there a way to assign tickets to specific agents in the shared inbox? I need this feature."
    ]
    
    results = []
    for email in test_emails:
        result = generate_reply(email, retriever)
        results.append(result)
        print(f"Generated reply for: {email[:40]}...\n")
        
    # Save to disk for the evaluation phase
    output_path = "data/generated_replies.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Successfully generated {len(results)} replies. Saved to {output_path}")

if __name__ == "__main__":
    main()