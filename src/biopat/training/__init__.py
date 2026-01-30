"""Training Module for BioPAT.

Provides utilities for domain adaptation and fine-tuning:
- Dense retriever fine-tuning with contrastive learning
- Cross-encoder reranker fine-tuning
- Knowledge distillation from large to small models
- SPLADE fine-tuning with FLOPS regularization

Example:
    ```python
    from biopat.training import create_dense_trainer, load_training_data

    # Load training data
    train_data = load_training_data("data/train.jsonl")

    # Create trainer
    trainer = create_dense_trainer(
        model_name="allenai/scibert_scivocab_uncased",
        epochs=3,
        learning_rate=2e-5,
    )

    # Fine-tune
    metrics = trainer.train(train_data)

    # Save
    trainer.save("models/fine_tuned_scibert")
    ```
"""

from biopat.training.domain_adaptation import (
    TrainingConfig,
    TrainingExample,
    ContrastiveDataset,
    DenseRetrieverTrainer,
    CrossEncoderTrainer,
    KnowledgeDistillation,
    create_dense_trainer,
    create_cross_encoder_trainer,
    load_training_data,
)

__all__ = [
    # Config
    "TrainingConfig",
    "TrainingExample",
    "ContrastiveDataset",

    # Trainers
    "DenseRetrieverTrainer",
    "CrossEncoderTrainer",
    "KnowledgeDistillation",

    # Factory functions
    "create_dense_trainer",
    "create_cross_encoder_trainer",
    "load_training_data",
]
