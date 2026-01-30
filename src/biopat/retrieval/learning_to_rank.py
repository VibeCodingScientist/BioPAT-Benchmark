"""Learning-to-Rank (LTR) Module for BioPAT.

Implements state-of-the-art learning-to-rank methods to optimally combine
multiple retrieval signals (dense, sparse, molecular, sequence scores).

Methods:
- LambdaMART: Gradient boosted trees with listwise optimization
- RankNet: Pairwise neural ranking
- ListNet: Listwise neural ranking with cross-entropy loss

Reference:
- Burges et al., "Learning to Rank using Gradient Descent" (RankNet)
- Cao et al., "Learning to Rank: From Pairwise Approach to Listwise Approach" (ListNet)
- Wu et al., "Adapting Boosting for Information Retrieval Measures" (LambdaMART)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@dataclass
class RankingFeatures:
    """Features for a query-document pair."""

    query_id: str
    doc_id: str

    # Retrieval scores from different methods
    bm25_score: float = 0.0
    dense_score: float = 0.0
    splade_score: float = 0.0
    colbert_score: float = 0.0
    reranker_score: float = 0.0

    # Molecular scores
    tanimoto_score: float = 0.0
    mol_embedding_score: float = 0.0

    # Sequence scores
    blast_score: float = 0.0
    seq_embedding_score: float = 0.0

    # Document features
    doc_length: int = 0
    claim_count: int = 0
    has_chemical: bool = False
    has_sequence: bool = False

    # Query features
    query_length: int = 0
    query_term_count: int = 0

    # Interaction features
    term_overlap: float = 0.0
    idf_weighted_overlap: float = 0.0

    # Label
    relevance: int = 0  # 0=irrelevant, 1=partial, 2=highly relevant

    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for ML models."""
        return np.array([
            self.bm25_score,
            self.dense_score,
            self.splade_score,
            self.colbert_score,
            self.reranker_score,
            self.tanimoto_score,
            self.mol_embedding_score,
            self.blast_score,
            self.seq_embedding_score,
            self.doc_length / 10000,  # Normalize
            self.claim_count / 100,
            float(self.has_chemical),
            float(self.has_sequence),
            self.query_length / 100,
            self.query_term_count / 20,
            self.term_overlap,
            self.idf_weighted_overlap,
        ])

    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for interpretability."""
        return [
            "bm25_score",
            "dense_score",
            "splade_score",
            "colbert_score",
            "reranker_score",
            "tanimoto_score",
            "mol_embedding_score",
            "blast_score",
            "seq_embedding_score",
            "doc_length_norm",
            "claim_count_norm",
            "has_chemical",
            "has_sequence",
            "query_length_norm",
            "query_term_count_norm",
            "term_overlap",
            "idf_weighted_overlap",
        ]


class BaseLTRModel(ABC):
    """Base class for learning-to-rank models."""

    @abstractmethod
    def fit(
        self,
        features: List[RankingFeatures],
        group_sizes: List[int],
    ) -> None:
        """Train the model.

        Args:
            features: List of RankingFeatures for all query-doc pairs
            group_sizes: Number of documents per query (for listwise methods)
        """
        pass

    @abstractmethod
    def predict(self, features: List[RankingFeatures]) -> np.ndarray:
        """Predict scores for query-document pairs.

        Args:
            features: Features for query-document pairs

        Returns:
            Predicted relevance scores
        """
        pass

    def rerank(
        self,
        query_id: str,
        doc_features: List[RankingFeatures],
    ) -> List[Tuple[str, float]]:
        """Rerank documents for a query.

        Args:
            query_id: Query identifier
            doc_features: Features for each candidate document

        Returns:
            List of (doc_id, score) sorted by predicted relevance
        """
        scores = self.predict(doc_features)
        ranked = sorted(
            zip([f.doc_id for f in doc_features], scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked


class LambdaMARTRanker(BaseLTRModel):
    """LambdaMART ranking using LightGBM.

    Gradient boosted trees with NDCG-optimized lambdas.
    State-of-the-art for feature-based ranking.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_data_in_leaf: int = 20,
        feature_fraction: float = 0.9,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        verbose: int = -1,
    ):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM required. Install with: pip install lightgbm")

        self.params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10, 20],
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "verbose": verbose,
            "force_row_wise": True,
        }
        self.model = None
        self.feature_importance_ = None

    def fit(
        self,
        features: List[RankingFeatures],
        group_sizes: List[int],
    ) -> None:
        """Train LambdaMART model."""
        X = np.array([f.to_feature_vector() for f in features])
        y = np.array([f.relevance for f in features])

        train_data = lgb.Dataset(
            X, label=y,
            group=group_sizes,
            feature_name=RankingFeatures.feature_names(),
        )

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params["n_estimators"],
        )

        # Store feature importance
        self.feature_importance_ = dict(zip(
            RankingFeatures.feature_names(),
            self.model.feature_importance(importance_type="gain")
        ))

        logger.info(f"LambdaMART trained. Top features: {self._top_features(5)}")

    def predict(self, features: List[RankingFeatures]) -> np.ndarray:
        """Predict relevance scores."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = np.array([f.to_feature_vector() for f in features])
        return self.model.predict(X)

    def _top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N features by importance."""
        if self.feature_importance_ is None:
            return []
        sorted_feats = sorted(
            self.feature_importance_.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_feats[:n]

    def save(self, path: str) -> None:
        """Save model to file."""
        if self.model is not None:
            self.model.save_model(path)

    def load(self, path: str) -> None:
        """Load model from file."""
        self.model = lgb.Booster(model_file=path)


class XGBoostRanker(BaseLTRModel):
    """XGBoost-based ranking with pairwise or listwise objectives."""

    def __init__(
        self,
        objective: str = "rank:ndcg",  # or "rank:pairwise"
        n_estimators: int = 300,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost required. Install with: pip install xgboost")

        self.params = {
            "objective": objective,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "eval_metric": "ndcg@10",
            "tree_method": "hist",
        }
        self.model = None

    def fit(
        self,
        features: List[RankingFeatures],
        group_sizes: List[int],
    ) -> None:
        """Train XGBoost ranker."""
        X = np.array([f.to_feature_vector() for f in features])
        y = np.array([f.relevance for f in features])

        self.model = xgb.XGBRanker(**self.params)
        self.model.fit(X, y, group=group_sizes)

        logger.info("XGBoost ranker trained.")

    def predict(self, features: List[RankingFeatures]) -> np.ndarray:
        """Predict relevance scores."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = np.array([f.to_feature_vector() for f in features])
        return self.model.predict(X)


class RankNetModel(BaseLTRModel):
    """RankNet: Pairwise neural ranking.

    Learns from pairs of documents where one is more relevant than another.
    Uses cross-entropy loss on pairwise preferences.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [64, 32],
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        dropout: float = 0.2,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, input_dim: int) -> nn.Module:
        """Build the scoring network."""
        layers = []
        prev_dim = input_dim

        for dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))

        return nn.Sequential(*layers)

    def fit(
        self,
        features: List[RankingFeatures],
        group_sizes: List[int],
    ) -> None:
        """Train RankNet model using pairwise loss."""
        X = np.array([f.to_feature_vector() for f in features])
        y = np.array([f.relevance for f in features])

        input_dim = X.shape[1]
        self.model = self._build_model(input_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Generate pairwise training data
        pairs = self._generate_pairs(X, y, group_sizes)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            np.random.shuffle(pairs)

            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]

                x_i = torch.tensor([p[0] for p in batch], dtype=torch.float32).to(self.device)
                x_j = torch.tensor([p[1] for p in batch], dtype=torch.float32).to(self.device)
                labels = torch.tensor([p[2] for p in batch], dtype=torch.float32).to(self.device)

                optimizer.zero_grad()

                s_i = self.model(x_i).squeeze()
                s_j = self.model(x_j).squeeze()

                # RankNet loss: cross-entropy on pairwise preferences
                loss = F.binary_cross_entropy_with_logits(s_i - s_j, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                logger.info(f"RankNet Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss:.4f}")

        logger.info("RankNet training complete.")

    def _generate_pairs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_sizes: List[int],
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Generate pairwise training data."""
        pairs = []
        idx = 0

        for size in group_sizes:
            group_X = X[idx:idx + size]
            group_y = y[idx:idx + size]

            for i in range(size):
                for j in range(i + 1, size):
                    if group_y[i] != group_y[j]:
                        if group_y[i] > group_y[j]:
                            pairs.append((group_X[i], group_X[j], 1.0))
                        else:
                            pairs.append((group_X[j], group_X[i], 1.0))

            idx += size

        return pairs

    def predict(self, features: List[RankingFeatures]) -> np.ndarray:
        """Predict relevance scores."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = torch.tensor(
            [f.to_feature_vector() for f in features],
            dtype=torch.float32
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            scores = self.model(X).squeeze().cpu().numpy()

        return scores


class ListNetModel(BaseLTRModel):
    """ListNet: Listwise neural ranking.

    Optimizes cross-entropy loss between predicted and true
    probability distributions over document rankings.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [64, 32],
        learning_rate: float = 0.001,
        epochs: int = 100,
        dropout: float = 0.2,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, input_dim: int) -> nn.Module:
        """Build the scoring network."""
        layers = []
        prev_dim = input_dim

        for dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))

        return nn.Sequential(*layers)

    def fit(
        self,
        features: List[RankingFeatures],
        group_sizes: List[int],
    ) -> None:
        """Train ListNet model."""
        X = np.array([f.to_feature_vector() for f in features])
        y = np.array([f.relevance for f in features], dtype=np.float32)

        input_dim = X.shape[1]
        self.model = self._build_model(input_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Group data by query
        groups = []
        idx = 0
        for size in group_sizes:
            groups.append((X[idx:idx + size], y[idx:idx + size]))
            idx += size

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            np.random.shuffle(groups)

            for group_X, group_y in groups:
                x_tensor = torch.tensor(group_X, dtype=torch.float32).to(self.device)
                y_tensor = torch.tensor(group_y, dtype=torch.float32).to(self.device)

                optimizer.zero_grad()

                # Predicted scores
                pred_scores = self.model(x_tensor).squeeze()

                # ListNet loss: KL divergence between probability distributions
                pred_probs = F.softmax(pred_scores, dim=0)
                true_probs = F.softmax(y_tensor, dim=0)

                loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-10))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                logger.info(f"ListNet Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss:.4f}")

        logger.info("ListNet training complete.")

    def predict(self, features: List[RankingFeatures]) -> np.ndarray:
        """Predict relevance scores."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = torch.tensor(
            [f.to_feature_vector() for f in features],
            dtype=torch.float32
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            scores = self.model(X).squeeze().cpu().numpy()

        return scores


class EnsembleLTR(BaseLTRModel):
    """Ensemble of multiple LTR models."""

    def __init__(
        self,
        models: List[BaseLTRModel],
        weights: Optional[List[float]] = None,
    ):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def fit(
        self,
        features: List[RankingFeatures],
        group_sizes: List[int],
    ) -> None:
        """Train all models in ensemble."""
        for model in self.models:
            model.fit(features, group_sizes)

    def predict(self, features: List[RankingFeatures]) -> np.ndarray:
        """Predict using weighted ensemble."""
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(features)
            predictions.append(pred * weight)

        return np.sum(predictions, axis=0)


class LTRFeatureExtractor:
    """Extract features for LTR from retrieval results."""

    def __init__(
        self,
        dense_retriever=None,
        sparse_retriever=None,
        splade_retriever=None,
        colbert_retriever=None,
        reranker=None,
        molecular_retriever=None,
        sequence_retriever=None,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.splade_retriever = splade_retriever
        self.colbert_retriever = colbert_retriever
        self.reranker = reranker
        self.molecular_retriever = molecular_retriever
        self.sequence_retriever = sequence_retriever

    def extract_features(
        self,
        query: str,
        doc_id: str,
        doc_text: str,
        query_smiles: Optional[str] = None,
        doc_smiles: Optional[str] = None,
        query_sequence: Optional[str] = None,
        doc_sequence: Optional[str] = None,
        relevance: int = 0,
    ) -> RankingFeatures:
        """Extract all features for a query-document pair."""
        features = RankingFeatures(
            query_id=query,
            doc_id=doc_id,
            relevance=relevance,
        )

        # Text retrieval scores
        if self.sparse_retriever:
            features.bm25_score = self.sparse_retriever.score(query, doc_text)

        if self.dense_retriever:
            features.dense_score = self.dense_retriever.score(query, doc_text)

        if self.splade_retriever:
            features.splade_score = self.splade_retriever.score(query, doc_text)

        if self.colbert_retriever:
            features.colbert_score = self.colbert_retriever.score(query, doc_text)

        if self.reranker:
            features.reranker_score = self.reranker.score(query, doc_text)

        # Molecular scores
        if self.molecular_retriever and query_smiles and doc_smiles:
            features.tanimoto_score = self.molecular_retriever.tanimoto_similarity(
                query_smiles, doc_smiles
            )
            features.mol_embedding_score = self.molecular_retriever.embedding_similarity(
                query_smiles, doc_smiles
            )

        # Sequence scores
        if self.sequence_retriever and query_sequence and doc_sequence:
            features.blast_score = self.sequence_retriever.blast_score(
                query_sequence, doc_sequence
            )
            features.seq_embedding_score = self.sequence_retriever.embedding_similarity(
                query_sequence, doc_sequence
            )

        # Document features
        features.doc_length = len(doc_text.split())
        features.has_chemical = doc_smiles is not None
        features.has_sequence = doc_sequence is not None

        # Query features
        features.query_length = len(query.split())
        features.query_term_count = len(set(query.lower().split()))

        # Interaction features
        query_terms = set(query.lower().split())
        doc_terms = set(doc_text.lower().split())
        if query_terms:
            features.term_overlap = len(query_terms & doc_terms) / len(query_terms)

        return features


def create_ltr_ranker(
    method: str = "lambdamart",
    **kwargs,
) -> BaseLTRModel:
    """Factory function for LTR models.

    Args:
        method: One of "lambdamart", "xgboost", "ranknet", "listnet", "ensemble"
        **kwargs: Additional arguments for the model

    Returns:
        Configured LTR model
    """
    if method == "lambdamart":
        return LambdaMARTRanker(**kwargs)
    elif method == "xgboost":
        return XGBoostRanker(**kwargs)
    elif method == "ranknet":
        return RankNetModel(**kwargs)
    elif method == "listnet":
        return ListNetModel(**kwargs)
    elif method == "ensemble":
        models = [
            LambdaMARTRanker(),
            RankNetModel(),
            ListNetModel(),
        ]
        return EnsembleLTR(models)
    else:
        raise ValueError(f"Unknown LTR method: {method}")
