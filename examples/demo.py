"""
Demo script showing how to use the duplicate question identification system.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_siamese_model
from src.evaluate import predict
from src.utils import data_generator
import pickle


def load_model_and_vocab(model_path='models/model.pkl.gz', vocab_path='vocab.pkl'):
    """Load trained model and vocabulary."""
    # Load vocabulary
    try:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Loaded vocabulary with {len(vocab)} tokens")
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {vocab_path}")
        print("Please ensure you have vocab.pkl in the project root.")
        sys.exit(1)
    
    # Create and load model
    try:
        model = create_siamese_model(vocab_size=len(vocab), d_model=128)
        model.init_from_file(model_path)
        print(f"Loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure you have trained model in the models/ directory.")
        sys.exit(1)
    
    return model, vocab


def demo_predictions(model, vocab, threshold=0.7):
    """Run demo predictions on example question pairs."""
    
    # Example question pairs
    examples = [
        {
            "q1": "How do I learn Python programming?",
            "q2": "What's the best way to learn Python?",
            "expected": "Likely Duplicate"
        },
        {
            "q1": "What is the capital of France?",
            "q2": "Who is the president of France?",
            "expected": "Not Duplicate"
        },
        {
            "q1": "How can I lose weight fast?",
            "q2": "What are quick ways to lose weight?",
            "expected": "Likely Duplicate"
        },
        {
            "q1": "What is machine learning?",
            "q2": "How does deep learning work?",
            "expected": "Not Duplicate"
        },
        {
            "q1": "When will I see you?",
            "q2": "When can I see you again?",
            "expected": "Likely Duplicate"
        }
    ]
    
    print("\n" + "="*80)
    print("DUPLICATE QUESTION IDENTIFICATION - DEMO")
    print("="*80)
    
    for i, example in enumerate(examples, 1):
        print(f"\n[Example {i}]")
        print(f"Question 1: {example['q1']}")
        print(f"Question 2: {example['q2']}")
        print(f"Expected:   {example['expected']}")
        print("-" * 80)
        
        is_duplicate, similarity = predict(
            example['q1'],
            example['q2'],
            threshold=threshold,
            model=model,
            vocab=vocab,
            data_generator=data_generator,
            verbose=False
        )
        
        result = "DUPLICATE" if is_duplicate else "NOT DUPLICATE"
        print(f"Prediction: {result}")
        print(f"Similarity Score: {similarity:.4f} (threshold: {threshold})")
    
    print("\n" + "="*80)


def interactive_mode(model, vocab, threshold=0.7):
    """Interactive mode for testing custom question pairs."""
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Enter question pairs to check for duplicates (or 'quit' to exit)\n")
    
    while True:
        try:
            q1 = input("Question 1: ").strip()
            if q1.lower() in ['quit', 'exit', 'q']:
                break
            
            q2 = input("Question 2: ").strip()
            if q2.lower() in ['quit', 'exit', 'q']:
                break
            
            if not q1 or not q2:
                print("Both questions must be non-empty. Please try again.\n")
                continue
            
            print("-" * 80)
            is_duplicate, similarity = predict(
                q1, q2,
                threshold=threshold,
                model=model,
                vocab=vocab,
                data_generator=data_generator,
                verbose=False
            )
            
            result = "DUPLICATE" if is_duplicate else "NOT DUPLICATE"
            print(f"Prediction: {result}")
            print(f"Similarity Score: {similarity:.4f}")
            print("="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Main demo function."""
    # Load model and vocabulary
    model, vocab = load_model_and_vocab()
    
    # Run demo predictions
    demo_predictions(model, vocab, threshold=0.7)
    
    # Ask if user wants to try interactive mode
    print("\nWould you like to try interactive mode? (y/n): ", end='')
    response = input().strip().lower()
    
    if response in ['y', 'yes']:
        interactive_mode(model, vocab, threshold=0.7)
    
    print("\nThank you for using the Duplicate Question Identification system!")


if __name__ == "__main__":
    main()

