import os
import json
from dotenv import load_dotenv
from groq import Groq

def evaluate_reply(item):
    """Prompts the LLM Judge to evaluate a single generated reply."""
    # Initialize client directly inside the function to avoid scope issues
    load_dotenv()
    client = Groq()
    
    incoming_email = item["incoming_email"]
    retrieved_context = item["retrieved_context"]
    generated_reply = item["generated_reply"]
    
    context_str = ""
    for i, ctx in enumerate(retrieved_context, 1):
        context_str += f"Example {i} Policy: {ctx['internal_policy']}\n"
        
    prompt = f"""You are an expert QA evaluator for a customer support AI. 
Your job is to grade an AI-generated reply to a customer email based on strict criteria.

--- INCOMING CUSTOMER EMAIL ---
{incoming_email}

--- AVAILABLE INTERNAL POLICIES (Retrieved Context) ---
{context_str}

--- AI GENERATED REPLY ---
{generated_reply}

--- EVALUATION CRITERIA ---
1. Policy Adherence (0-5): Did the reply strictly follow the internal policies provided? (0 = completely ignored/hallucinated, 5 = perfectly followed).
2. Tone & Empathy (0-5): Is the reply polite, professional, and empathetic? (0 = rude/robotic, 5 = highly empathetic and human-like).
3. Completeness & Clarity (0-5): Did it directly answer the customer's question concisely? (0 = ignored question/rambling, 5 = perfect, concise answer).

Output ONLY a valid JSON object with this exact structure:
{{
  "policy_adherence": {{"score": <int>, "reason": "<1 sentence explanation>"}},
  "tone_empathy": {{"score": <int>, "reason": "<1 sentence explanation>"}},
  "completeness_clarity": {{"score": <int>, "reason": "<1 sentence explanation>"}},
  "overall_score": <float>,
  "overall_reasoning": "<1-2 sentences summary>"
}}
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1 
    )
    
    eval_result = json.loads(response.choices[0].message.content)
    
    # Calculate weighted score directly here so it's always returned
    weighted = (eval_result["policy_adherence"]["score"] * 0.4 + 
                eval_result["tone_empathy"]["score"] * 0.3 + 
                eval_result["completeness_clarity"]["score"] * 0.3)
    eval_result["calculated_weighted_score"] = round(weighted, 2)
    
    return eval_result

def main():
    with open("data/generated_replies.json", "r") as f:
        generated_replies = json.load(f)
        
    results = []
    total_scores = {"policy_adherence": 0, "tone_empathy": 0, "completeness_clarity": 0, "overall_score": 0}
    
    print("Starting evaluation via Groq (Llama 3.3 Judge)...")
    for i, item in enumerate(generated_replies):
        print(f"Evaluating reply {i+1}/{len(generated_replies)}...")
        eval_result = evaluate_reply(item)
        
        results.append({
            "incoming_email": item["incoming_email"],
            "generated_reply": item["generated_reply"],
            "evaluation": eval_result
        })
        
        for key in total_scores:
            if key == "overall_score":
                total_scores[key] += eval_result["calculated_weighted_score"]
            else:
                total_scores[key] += eval_result[key]["score"]
                
    n = len(results)
    averages = {k: round(v / n, 2) for k, v in total_scores.items()}
    
    final_output = {
        "per_response_evaluations": results,
        "overall_system_scores": averages
    }
    
    with open("data/evaluation_results.json", "w") as f:
        json.dump(final_output, f, indent=2)
        
    print("\n--- OVERALL SYSTEM SCORES ---")
    for k, v in averages.items():
        print(f"{k}: {v}")
    print(f"\nDetailed results saved to data/evaluation_results.json")

if __name__ == "__main__":
    main()