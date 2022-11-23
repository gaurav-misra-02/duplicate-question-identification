"""
Data preprocessing and vocabulary building functions.
"""

import numpy as np
import pandas as pd
import nltk
from collections import defaultdict
from typing import Tuple, Dict


def load_data(
    filepath: str,
    n_train: int = 300000,
    n_test: int = 10240
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split the question pairs dataset.
    
    Args:
        filepath: Path to the CSV file containing question pairs
        n_train: Number of samples for training
        n_test: Number of samples for testing
    
    Returns:
        Tuple of (train_data, test_data) as pandas DataFrames
    """
    data = pd.read_csv(filepath)
    data_train = data[:n_train]
    data_test = data[n_train:n_train + n_test]
    return data_train, data_test


def extract_duplicate_pairs(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract only duplicate question pairs from the dataset.
    
    For training, we use only duplicate pairs and leverage the batch structure
    to generate negative examples (non-duplicates) implicitly.
    
    Args:
        data: DataFrame containing question pairs and duplicate labels
    
    Returns:
        Tuple of (Q1_words, Q2_words) containing duplicate question pairs
    """
    duplicate_indices = (data['is_duplicate'] == 1).to_numpy()
    duplicate_indices = [i for i, x in enumerate(duplicate_indices) if x]
    
    Q1_words = np.array(data['question1'][duplicate_indices])
    Q2_words = np.array(data['question2'][duplicate_indices])
    
    return Q1_words, Q2_words


def build_vocabulary(Q1_train_words: np.ndarray, Q2_train_words: np.ndarray) -> Dict:
    """
    Build vocabulary from training questions.
    
    Tokenizes questions and assigns a unique integer to each word.
    Words not in vocabulary will map to 0 (OOV token).
    
    Args:
        Q1_train_words: Array of question strings (first set)
        Q2_train_words: Array of question strings (second set)
    
    Returns:
        Dictionary mapping words to integer IDs
    """
    nltk.download('punkt', quiet=True)
    
    vocab = defaultdict(lambda: 0)
    vocab['<PAD>'] = 1
    
    Q1_train = np.empty_like(Q1_train_words)
    Q2_train = np.empty_like(Q2_train_words)
    
    for index in range(len(Q1_train_words)):
        Q1_train[index] = nltk.word_tokenize(Q1_train_words[index])
        Q2_train[index] = nltk.word_tokenize(Q2_train_words[index])
        
        combined_tokens = Q1_train[index] + Q2_train[index]
        for word in combined_tokens:
            if word not in vocab:
                vocab[word] = len(vocab) + 1
    
    return vocab, Q1_train, Q2_train


def tokenize_questions(Q_words: np.ndarray) -> np.ndarray:
    """
    Tokenize questions into word tokens.
    
    Args:
        Q_words: Array of question strings
    
    Returns:
        Array of tokenized questions
    """
    Q_tokens = np.empty_like(Q_words)
    for index in range(len(Q_words)):
        Q_tokens[index] = nltk.word_tokenize(Q_words[index])
    return Q_tokens


def encode_questions(Q_tokens: np.ndarray, vocab: Dict) -> np.ndarray:
    """
    Encode tokenized questions as sequences of vocabulary indices.
    
    Args:
        Q_tokens: Array of tokenized questions
        vocab: Vocabulary dictionary mapping words to IDs
    
    Returns:
        Array of encoded questions
    """
    Q_encoded = np.empty_like(Q_tokens)
    for i in range(len(Q_tokens)):
        Q_encoded[i] = [vocab[word] for word in Q_tokens[i]]
    return Q_encoded


def prepare_data(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    train_split: float = 0.8
) -> Tuple:
    """
    Complete data preparation pipeline.
    
    Args:
        data_train: Training DataFrame
        data_test: Testing DataFrame
        train_split: Proportion of training data to use for training (rest for validation)
    
    Returns:
        Tuple containing:
            - vocab: Vocabulary dictionary
            - train_Q1, train_Q2: Training question pairs (encoded)
            - val_Q1, val_Q2: Validation question pairs (encoded)
            - Q1_test, Q2_test: Test question pairs (encoded)
            - y_test: Test labels
    """
    # Extract duplicate pairs for training
    Q1_train_words, Q2_train_words = extract_duplicate_pairs(data_train)
    
    # Build vocabulary and tokenize training data
    vocab, Q1_train_tokens, Q2_train_tokens = build_vocabulary(
        Q1_train_words, Q2_train_words
    )
    
    # Encode training data
    Q1_train = encode_questions(Q1_train_tokens, vocab)
    Q2_train = encode_questions(Q2_train_tokens, vocab)
    
    # Prepare test data
    Q1_test_words = np.array(data_test['question1'])
    Q2_test_words = np.array(data_test['question2'])
    y_test = np.array(data_test['is_duplicate'])
    
    Q1_test_tokens = tokenize_questions(Q1_test_words)
    Q2_test_tokens = tokenize_questions(Q2_test_words)
    
    Q1_test = encode_questions(Q1_test_tokens, vocab)
    Q2_test = encode_questions(Q2_test_tokens, vocab)
    
    # Split training data into train and validation
    cut_off = int(len(Q1_train) * train_split)
    train_Q1, train_Q2 = Q1_train[:cut_off], Q2_train[:cut_off]
    val_Q1, val_Q2 = Q1_train[cut_off:], Q2_train[cut_off:]
    
    return vocab, train_Q1, train_Q2, val_Q1, val_Q2, Q1_test, Q2_test, y_test

