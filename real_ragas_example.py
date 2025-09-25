"""
Simple RAGAS Example - Just Dataset and Scores
==============================================

This file creates a simple dataset and runs RAGAS evaluation to get scores.
"""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, answer_similarity
from datasets import Dataset
from utils import extract_ragas_scores

def main():
    # Simple test dataset
    data = [
        {
            "question": "How does chemotherapy work?",
            "answer": "The weather is sunny today and I like pizza",
            "contexts": ["Chemotherapy uses drugs to kill cancer cells."],
            "ground_truth": "Chemotherapy uses drugs to kill cancer cells."
        },
        {
            "question": "What are diabetes symptoms?",
            "answer": "12345 67890 random words nonsense",
            "contexts": ["Diabetes symptoms include thirst and frequent urination."],
            "ground_truth": "Diabetes symptoms include thirst and frequent urination."
        }
    ]
    
    # Convert to dataset
    dataset = Dataset.from_list(data)
    
    print("Running RAGAS evaluation...")
    
    # Run evaluation
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, answer_similarity]
    )
    
    # Extract scores using shared utility function
    scores = extract_ragas_scores(results)

if __name__ == "__main__":
    main()




