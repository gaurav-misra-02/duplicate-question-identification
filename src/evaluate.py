"""
Evaluation and prediction functions for the Siamese network.
"""

import numpy as np
import nltk
from trax.fastmath import numpy as fastnp
from typing import Dict, Callable, Tuple


def classify(
    test_Q1: np.ndarray,
    test_Q2: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    model,
    vocab: Dict,
    data_generator: Callable,
    batch_size: int = 64
) -> float:
    """
    Evaluate model accuracy on a test set.
    
    Computes cosine similarity between question pairs and classifies them as
    duplicates or not based on a threshold.
    
    Args:
        test_Q1: First set of test questions (encoded)
        test_Q2: Second set of test questions (encoded)
        labels: Ground truth labels (1 for duplicate, 0 for not duplicate)
        threshold: Similarity threshold for classification
        model: Trained Siamese model
        vocab: Vocabulary dictionary
        data_generator: Data generator function
        batch_size: Batch size for evaluation
    
    Returns:
        Classification accuracy
    """
    accuracy = 0
    
    for i in range(0, len(test_Q1), batch_size):
        # Get batch
        q1_batch, q2_batch = next(data_generator(
            test_Q1[i:i + batch_size],
            test_Q2[i:i + batch_size],
            batch_size,
            vocab['<PAD>'],
            shuffle=False
        ))
        labels_batch = labels[i:i + batch_size]
        
        # Get embeddings
        embeddings_1, embeddings_2 = model([q1_batch, q2_batch])
        
        # Compute similarities and classify
        for j in range(batch_size):
            similarity = fastnp.dot(embeddings_1[j], embeddings_2[j])
            prediction = similarity > threshold
            accuracy += (labels_batch[j] == prediction)
    
    return accuracy / len(test_Q1)


def predict(
    question1: str,
    question2: str,
    threshold: float,
    model,
    vocab: Dict,
    data_generator: Callable,
    verbose: bool = False
) -> Tuple[bool, float]:
    """
    Predict whether two questions are duplicates.
    
    Args:
        question1: First question (string)
        question2: Second question (string)
        threshold: Similarity threshold for classification
        model: Trained Siamese model
        vocab: Vocabulary dictionary
        data_generator: Data generator function
        verbose: Whether to print detailed information
    
    Returns:
        Tuple of (is_duplicate, similarity_score)
    """
    # Tokenize questions
    q1_tokens = nltk.word_tokenize(question1)
    q2_tokens = nltk.word_tokenize(question2)
    
    # Encode questions (OOV words get 0)
    Q1_encoded = [vocab.get(word, 0) for word in q1_tokens]
    Q2_encoded = [vocab.get(word, 0) for word in q2_tokens]
    
    # Pad questions
    Q1_padded, Q2_padded = next(data_generator(
        [Q1_encoded], [Q2_encoded], 1, vocab['<PAD>'], shuffle=False
    ))
    
    # Get embeddings and compute similarity
    embeddings_1, embeddings_2 = model([Q1_padded, Q2_padded])
    similarity = float(fastnp.dot(embeddings_1, embeddings_2.T)[0][0])
    is_duplicate = similarity > threshold
    
    if verbose:
        print(f"Question 1: {question1}")
        print(f"Question 2: {question2}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Threshold: {threshold}")
        print(f"Prediction: {'Duplicate' if is_duplicate else 'Not Duplicate'}")
    
    return is_duplicate, similarity

