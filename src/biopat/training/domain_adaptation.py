"""Domain Adaptation and Fine-tuning for BioPAT.

Provides utilities for adapting pre-trained models to the
biomedical patent domain:

- Contrastive fine-tuning for dense retrievers
- SPLADE fine-tuning with FLOPS regularization
- Cross-encoder fine-tuning for rerankers
- Knowledge distillation from large to small models

Reference:
- Izacard et al., "Unsupervised Dense Information Retrieval with Contrastive Learning" (2021)
- Formal et al., "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking" (2021)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import transformers
try:
    from transformers import (
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""

    # Model
    model_name: str = "allenai/scibert_scivocab_uncased"
    max_length: int = 256

    # Training
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1

    # Loss
    temperature: float = 0.05  # For contrastive loss
    hard_negative_weight: float = 1.0

    # Regularization
    splade_flops_weight: float = 0.0001  # SPLADE regularization

    # Output
    output_dir: str = "models/fine_tuned"
    save_steps: int = 1000
    eval_steps: int = 500

    # Hardware
    fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingExample:
    """A training example for retrieval fine-tuning."""

    query: str
    positive: str  # Relevant document
    negatives: List[str] = field(default_factory=list)  # Hard negatives

    # Optional metadata
    query_id: Optional[str] = None
    positive_id: Optional[str] = None
    negative_ids: List[str] = field(default_factory=list)


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning."""

    def __init__(
        self,
        examples: List[TrainingExample],
        tokenizer,
        max_length: int = 256,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokenize query
        query_encoding = self.tokenizer(
            example.query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize positive
        positive_encoding = self.tokenizer(
            example.positive,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize negatives
        negative_encodings = []
        for neg in example.negatives:
            neg_encoding = self.tokenizer(
                neg,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            negative_encodings.append(neg_encoding)

        return {
            "query_input_ids": query_encoding["input_ids"].squeeze(0),
            "query_attention_mask": query_encoding["attention_mask"].squeeze(0),
            "positive_input_ids": positive_encoding["input_ids"].squeeze(0),
            "positive_attention_mask": positive_encoding["attention_mask"].squeeze(0),
            "negative_input_ids": torch.stack([e["input_ids"].squeeze(0) for e in negative_encodings]) if negative_encodings else torch.tensor([]),
            "negative_attention_masks": torch.stack([e["attention_mask"].squeeze(0) for e in negative_encodings]) if negative_encodings else torch.tensor([]),
        }


class DenseRetrieverTrainer:
    """Fine-tune dense retrieval models with contrastive learning.

    Uses in-batch negatives plus hard negatives for training.
    """

    def __init__(
        self,
        config: TrainingConfig,
    ):
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "PyTorch and transformers required. "
                "Install with: pip install torch transformers"
            )

        self.config = config
        self.device = torch.device(config.device)

        # Load model and tokenizer
        logger.info(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name).to(self.device)

        # Optimizer
        self.optimizer = None
        self.scheduler = None

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pooling over tokens."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text to embeddings."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        return F.normalize(embeddings, p=2, dim=1)

    def _contrastive_loss(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute contrastive loss with in-batch negatives."""
        batch_size = query_embeddings.shape[0]

        # Positive scores
        positive_scores = torch.sum(query_embeddings * positive_embeddings, dim=1)

        # In-batch negative scores
        # Each query uses all other positives in the batch as negatives
        all_scores = torch.matmul(query_embeddings, positive_embeddings.T)

        # Add hard negatives if provided
        if negative_embeddings is not None and negative_embeddings.numel() > 0:
            hard_neg_scores = torch.matmul(
                query_embeddings.unsqueeze(1),
                negative_embeddings.transpose(1, 2)
            ).squeeze(1)
            all_scores = torch.cat([all_scores, hard_neg_scores], dim=1)

        # Scale by temperature
        all_scores = all_scores / self.config.temperature

        # Labels: positive is at position i for query i
        labels = torch.arange(batch_size, device=self.device)

        # Cross-entropy loss
        loss = F.cross_entropy(all_scores, labels)

        return loss

    def train(
        self,
        train_examples: List[TrainingExample],
        val_examples: Optional[List[TrainingExample]] = None,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the model.

        Args:
            train_examples: Training examples
            val_examples: Optional validation examples

        Returns:
            Dict with training metrics
        """
        # Create dataset and dataloader
        train_dataset = ContrastiveDataset(
            train_examples,
            self.tokenizer,
            self.config.max_length,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Setup optimizer
        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        metrics = {"train_loss": [], "val_loss": []}
        self.model.train()

        global_step = 0
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                query_ids = batch["query_input_ids"].to(self.device)
                query_mask = batch["query_attention_mask"].to(self.device)
                pos_ids = batch["positive_input_ids"].to(self.device)
                pos_mask = batch["positive_attention_mask"].to(self.device)

                # Encode
                query_emb = self._encode(query_ids, query_mask)
                pos_emb = self._encode(pos_ids, pos_mask)

                # Handle hard negatives
                neg_emb = None
                if batch["negative_input_ids"].numel() > 0:
                    neg_ids = batch["negative_input_ids"].to(self.device)
                    neg_mask = batch["negative_attention_masks"].to(self.device)
                    # Reshape for batch processing
                    bs, num_neg, seq_len = neg_ids.shape
                    neg_ids_flat = neg_ids.view(-1, seq_len)
                    neg_mask_flat = neg_mask.view(-1, seq_len)
                    neg_emb_flat = self._encode(neg_ids_flat, neg_mask_flat)
                    neg_emb = neg_emb_flat.view(bs, num_neg, -1)

                # Compute loss
                loss = self._contrastive_loss(query_emb, pos_emb, neg_emb)

                # Backward
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item() * self.config.gradient_accumulation_steps

                # Logging
                if global_step % 100 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{self.config.epochs}, "
                        f"Step {global_step}, Loss: {loss.item():.4f}"
                    )

            avg_loss = epoch_loss / len(train_loader)
            metrics["train_loss"].append(avg_loss)
            logger.info(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}")

            # Validation
            if val_examples:
                val_loss = self._evaluate(val_examples)
                metrics["val_loss"].append(val_loss)
                logger.info(f"Validation Loss: {val_loss:.4f}")

        return metrics

    def _evaluate(self, examples: List[TrainingExample]) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        dataset = ContrastiveDataset(examples, self.tokenizer, self.config.max_length)
        loader = DataLoader(dataset, batch_size=self.config.batch_size)

        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                query_ids = batch["query_input_ids"].to(self.device)
                query_mask = batch["query_attention_mask"].to(self.device)
                pos_ids = batch["positive_input_ids"].to(self.device)
                pos_mask = batch["positive_attention_mask"].to(self.device)

                query_emb = self._encode(query_ids, query_mask)
                pos_emb = self._encode(pos_ids, pos_mask)

                loss = self._contrastive_loss(query_emb, pos_emb)
                total_loss += loss.item()

        self.model.train()
        return total_loss / len(loader)

    def save(self, output_dir: Optional[str] = None):
        """Save the fine-tuned model."""
        output_dir = output_dir or self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save config
        with open(Path(output_dir) / "training_config.json", "w") as f:
            json.dump(vars(self.config), f, indent=2)

        logger.info(f"Model saved to {output_dir}")


class CrossEncoderTrainer:
    """Fine-tune cross-encoder rerankers."""

    def __init__(
        self,
        config: TrainingConfig,
    ):
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "PyTorch and transformers required."
            )

        self.config = config
        self.device = torch.device(config.device)

        # Load model
        logger.info(f"Loading cross-encoder: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=1,
        ).to(self.device)

    def train(
        self,
        train_examples: List[Tuple[str, str, float]],  # (query, doc, label)
        val_examples: Optional[List[Tuple[str, str, float]]] = None,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the cross-encoder.

        Args:
            train_examples: List of (query, document, relevance_label)
            val_examples: Optional validation examples

        Returns:
            Training metrics
        """
        # Prepare data
        train_encodings = self.tokenizer(
            [(ex[0], ex[1]) for ex in train_examples],
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        train_labels = torch.tensor([ex[2] for ex in train_examples], dtype=torch.float)

        # Create dataloader
        dataset = torch.utils.data.TensorDataset(
            train_encodings["input_ids"],
            train_encodings["attention_mask"],
            train_labels,
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        # Optimizer
        total_steps = len(loader) * self.config.epochs
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps,
        )

        # Training
        metrics = {"train_loss": []}
        self.model.train()

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0

            for batch in loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits.squeeze(-1)

                loss = F.mse_loss(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            metrics["train_loss"].append(avg_loss)
            logger.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        return metrics

    def save(self, output_dir: Optional[str] = None):
        """Save the model."""
        output_dir = output_dir or self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Cross-encoder saved to {output_dir}")


class KnowledgeDistillation:
    """Distill knowledge from large teacher to small student model."""

    def __init__(
        self,
        teacher_model_name: str,
        student_model_name: str,
        temperature: float = 4.0,
        alpha: float = 0.5,  # Weight for distillation vs task loss
    ):
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError("PyTorch and transformers required.")

        self.temperature = temperature
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load models
        logger.info(f"Loading teacher: {teacher_model_name}")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        self.teacher_model = AutoModel.from_pretrained(teacher_model_name).to(self.device)
        self.teacher_model.eval()

        logger.info(f"Loading student: {student_model_name}")
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        self.student_model = AutoModel.from_pretrained(student_model_name).to(self.device)

    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss using KL divergence."""
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean")
        return loss * (self.temperature ** 2)

    def distill(
        self,
        texts: List[str],
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 5e-5,
    ) -> Dict[str, List[float]]:
        """
        Distill teacher knowledge to student.

        Args:
            texts: Training texts
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Training metrics
        """
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
        )

        metrics = {"distill_loss": []}
        self.student_model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Get teacher embeddings
                with torch.no_grad():
                    teacher_enc = self.teacher_tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=256,
                        return_tensors="pt",
                    ).to(self.device)

                    teacher_out = self.teacher_model(**teacher_enc)
                    teacher_emb = teacher_out.last_hidden_state[:, 0, :]

                # Get student embeddings
                student_enc = self.student_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(self.device)

                student_out = self.student_model(**student_enc)
                student_emb = student_out.last_hidden_state[:, 0, :]

                # Compute loss (MSE between embeddings)
                loss = F.mse_loss(student_emb, teacher_emb)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / (len(texts) // batch_size)
            metrics["distill_loss"].append(avg_loss)
            logger.info(f"Distillation Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        return metrics

    def save_student(self, output_dir: str):
        """Save the distilled student model."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.student_model.save_pretrained(output_dir)
        self.student_tokenizer.save_pretrained(output_dir)
        logger.info(f"Student model saved to {output_dir}")


def create_dense_trainer(
    model_name: str = "allenai/scibert_scivocab_uncased",
    **kwargs,
) -> DenseRetrieverTrainer:
    """Factory function for dense retriever trainer."""
    config = TrainingConfig(model_name=model_name, **kwargs)
    return DenseRetrieverTrainer(config)


def create_cross_encoder_trainer(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    **kwargs,
) -> CrossEncoderTrainer:
    """Factory function for cross-encoder trainer."""
    config = TrainingConfig(model_name=model_name, **kwargs)
    return CrossEncoderTrainer(config)


def load_training_data(
    data_path: Union[str, Path],
) -> List[TrainingExample]:
    """
    Load training data from JSONL file.

    Expected format:
    {"query": "...", "positive": "...", "negatives": ["...", "..."]}

    Args:
        data_path: Path to JSONL file

    Returns:
        List of TrainingExample
    """
    examples = []
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)
            examples.append(TrainingExample(
                query=data["query"],
                positive=data["positive"],
                negatives=data.get("negatives", []),
                query_id=data.get("query_id"),
                positive_id=data.get("positive_id"),
            ))
    return examples
