"""
Training utilities for the Siamese network.
"""

import os
import trax
from trax.supervised import training
from typing import Callable


def train_model(
    model_fn: Callable,
    loss_fn: Callable,
    train_generator,
    val_generator,
    output_dir: str = 'model/',
    learning_rate: float = 0.01,
    warmup_steps: int = 400
):
    """
    Train the Siamese model with triplet loss.
    
    Uses Adam optimizer with learning rate warmup and inverse square root decay
    for stable training.
    
    Args:
        model_fn: Function that returns the Siamese model
        loss_fn: Function that returns the loss layer
        train_generator: Data generator for training
        val_generator: Data generator for validation
        output_dir: Directory to save model checkpoints
        learning_rate: Peak learning rate after warmup
        warmup_steps: Number of warmup steps for learning rate schedule
    
    Returns:
        Training loop object
    """
    output_dir = os.path.expanduser(output_dir)
    
    train_task = training.TrainTask(
        labeled_data=train_generator,
        loss_layer=loss_fn(),
        optimizer=trax.optimizers.Adam(learning_rate),
        lr_schedule=trax.lr.warmup_and_rsqrt_decay(warmup_steps, learning_rate)
    )
    
    eval_task = training.EvalTask(
        labeled_data=val_generator,
        metrics=[loss_fn()],
    )
    
    training_loop = training.Loop(
        model_fn(),
        train_task,
        eval_tasks=[eval_task],
        output_dir=output_dir
    )
    
    return training_loop

