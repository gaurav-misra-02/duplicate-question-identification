"""
Siamese network model architecture using Trax.
"""

import trax
from trax import layers as tl
from trax.fastmath import numpy as fastnp
from functools import partial


def l2_normalize(x):
    """
    Normalize vectors to unit L2 norm.
    
    Args:
        x: Input vectors
    
    Returns:
        L2-normalized vectors
    """
    return x / fastnp.sqrt(fastnp.sum(x * x, axis=-1, keepdims=True))


def create_siamese_model(vocab_size: int = 41699, d_model: int = 128, mode: str = 'train'):
    """
    Create a Siamese network for question similarity.
    
    Architecture:
        - Embedding layer: Maps tokens to dense vectors
        - LSTM layer: Processes sequential information
        - Mean pooling: Aggregates sequence into fixed-size vector
        - L2 normalization: Normalizes vectors for cosine similarity
    
    The model processes two questions in parallel using shared weights,
    producing normalized embeddings that can be compared via cosine similarity.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of embeddings and LSTM hidden state
        mode: One of 'train', 'eval', or 'predict'
    
    Returns:
        Trax Parallel layer combining two identical question processors
    """
    question_processor = tl.Serial(
        tl.Embedding(vocab_size, d_model),
        tl.LSTM(d_model),
        tl.Mean(axis=1),
        tl.Fn('Normalize', lambda x: l2_normalize(x)),
    )
    
    # Process both questions in parallel with shared weights
    model = tl.Parallel(question_processor, question_processor)
    return model


def triplet_loss_fn(embeddings_1: fastnp.ndarray, embeddings_2: fastnp.ndarray, margin: float = 0.25) -> float:
    """
    Compute triplet loss with hard negative mining.
    
    This loss function encourages the model to produce similar embeddings for
    duplicate questions and dissimilar embeddings for non-duplicates. It uses
    both the closest negative and mean negative for robust training.
    
    Loss formulation:
        L1 = max(0, margin - cos(Q1, Q2_duplicate) + closest_negative)
        L2 = max(0, margin - cos(Q1, Q2_duplicate) + mean_negative)
        L = mean(L1 + L2)
    
    Args:
        embeddings_1: Embeddings for first set of questions (batch_size, d_model)
        embeddings_2: Embeddings for second set of questions (batch_size, d_model)
        margin: Margin for triplet loss (default: 0.25)
    
    Returns:
        Scalar loss value
    """
    # Compute pairwise cosine similarities
    scores = fastnp.dot(embeddings_1, embeddings_2.T)
    batch_size = len(scores)
    
    # Extract positive pairs (diagonal elements)
    positive = fastnp.diagonal(scores)
    
    # Find closest negative (hardest negative example)
    # Mask out positive pairs by subtracting large value from diagonal
    negative_without_positive = scores - fastnp.eye(batch_size) * 2.0
    closest_negative = negative_without_positive.max(axis=1)
    
    # Compute mean negative
    negative_zero_on_duplicate = (1.0 - fastnp.eye(batch_size)) * scores
    mean_negative = fastnp.sum(negative_zero_on_duplicate, axis=1) / (batch_size - 1)
    
    # Compute losses
    triplet_loss1 = fastnp.maximum(margin - positive + closest_negative, 0)
    triplet_loss2 = fastnp.maximum(margin - positive + mean_negative, 0)
    
    return fastnp.mean(triplet_loss1 + triplet_loss2)


def create_triplet_loss(margin: float = 0.25):
    """
    Create a Trax-compatible triplet loss layer.
    
    Args:
        margin: Margin for triplet loss
    
    Returns:
        Trax Fn layer wrapping the triplet loss function
    """
    triplet_loss = partial(triplet_loss_fn, margin=margin)
    return tl.Fn('TripletLoss', triplet_loss)

