import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluator import evaluate_reply

def main():
    print("--- RUNNING METRIC VALIDATION (SANITY CHECK) ---")
    
    # A deliberately terrible, hallucinated, and rude reply
    terrible_item = {
        "incoming_email": "How do I reset my password?",
        "retrieved_context": [{"internal_policy": "Always provide the official password reset link: hiverhq.com/reset"}],
        "generated_reply": "I don't care about your password. Just delete your account. Also, we are giving you a free $1000 credit for your trouble."
    }
    
    print("Feeding deliberately terrible reply to the Judge...")
    result = evaluate_reply(terrible_item)
    
    print("\nJUDGE RESULTS:")
    print(f"Policy Adherence: {result['policy_adherence']['score']}/5 - {result['policy_adherence']['reason']}")
    print(f"Tone & Empathy: {result['tone_empathy']['score']}/5 - {result['tone_empathy']['reason']}")
    print(f"Completeness: {result['completeness_clarity']['score']}/5 - {result['completeness_clarity']['reason']}")
    print(f"Overall Weighted Score: {result['calculated_weighted_score']}")
    
    if result['calculated_weighted_score'] < 2.0:
        print("\nVALIDATION PASSED: The Judge correctly penalized the terrible reply.")
    else:
        print("\nVALIDATION FAILED: The Judge gave a high score to a bad reply.")

if __name__ == "__main__":
    main()