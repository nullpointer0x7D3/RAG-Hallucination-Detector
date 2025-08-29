"""
Simple RAGAS Example - Just Dataset and Scores
==============================================

This file creates a simple dataset and runs RAGAS evaluation to get scores.
"""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, answer_similarity
from datasets import Dataset

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
    
    # Output scores
    print("\nRAGAS Scores:")
    
    
    # Handle case where results might be lists
    faithfulness_score = results['faithfulness']
    if isinstance(faithfulness_score, list):
        faithfulness_score = faithfulness_score[0] if faithfulness_score else 0
    
    answer_relevancy_score = results['answer_relevancy']
    if isinstance(answer_relevancy_score, list):
        answer_relevancy_score = answer_relevancy_score[0] if answer_relevancy_score else 0
    
    context_recall_score = results['context_recall']
    if isinstance(context_recall_score, list):
        context_recall_score = context_recall_score[0] if context_recall_score else 0

    answer_similarity_score = results['answer_similarity']
    if isinstance(answer_similarity_score, list):
        answer_similarity_score = answer_similarity_score[0] if answer_similarity_score else 0
    
    print(f"Faithfulness: {faithfulness_score:.3f}")
    print(f"Answer Relevancy: {answer_relevancy_score:.3f}")
    print(f"Context Recall: {context_recall_score:.3f}")
    print(f"Answer Similarity: {answer_similarity_score:.3f}")

if __name__ == "__main__":
    main()




