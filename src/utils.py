"""
Utility functions for data generation and preprocessing.
"""

import numpy as np
import random as rnd
from typing import Tuple, List, Generator


def data_generator(
    Q1: np.ndarray,
    Q2: np.ndarray,
    batch_size: int,
    pad: int = 1,
    shuffle: bool = True
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate batches of question pairs for training or evaluation.
    
    This generator yields padded batches of question pairs. Questions are padded
    to the nearest power of 2 for computational efficiency.
    
    Args:
        Q1: Array of tokenized questions (first set)
        Q2: Array of tokenized questions (second set)
        batch_size: Number of question pairs per batch
        pad: Padding token ID (default: 1)
        shuffle: Whether to shuffle the data (default: True)
    
    Yields:
        Tuple of (batch_Q1, batch_Q2) as numpy arrays with shape (batch_size, max_len)
    """
    input1 = []
    input2 = []
    idx = 0
    len_q = len(Q1)
    question_indexes = list(range(len_q))
    
    if shuffle:
        rnd.shuffle(question_indexes)
    
    while True:
        if idx >= len_q:
            idx = 0
            if shuffle:
                rnd.shuffle(question_indexes)
        
        q1 = Q1[question_indexes[idx]]
        q2 = Q2[question_indexes[idx]]
        
        idx += 1
        input1.append(q1)
        input2.append(q2)
        
        if len(input1) == batch_size:
            # Calculate maximum length and pad to nearest power of 2
            max_len = max(
                max([len(q) for q in input1]),
                max([len(q) for q in input2])
            )
            max_len = 2 ** int(np.ceil(np.log2(max_len)))
            
            # Pad questions to max_len
            b1 = []
            b2 = []
            for q1, q2 in zip(input1, input2):
                q1_padded = q1 + [pad] * (max_len - len(q1))
                q2_padded = q2 + [pad] * (max_len - len(q2))
                b1.append(q1_padded)
                b2.append(q2_padded)
            
            yield np.array(b1), np.array(b2)
            
            # Reset batches
            input1, input2 = [], []


def set_random_seed(seed: int = 34) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    rnd.seed(seed)
    np.random.seed(seed)

