# BioPAT API Reference

This document provides API documentation for BioPAT modules.

## Data Acquisition

### DataAcquisition

Unified interface for fetching data from all sources.

```python
from biopat.data import DataAcquisition, AcquisitionConfig

config = AcquisitionConfig(
    pubmed_api_key="your_key",
    patentsview_api_key="your_key",
    cache_dir=Path("data/.cache"),
)

async with DataAcquisition(config=config) as acq:
    # Fetch articles
    articles = await acq.fetch_pubmed("query", limit=100)
    preprints = await acq.fetch_biorxiv("query", limit=50)
    
    # Fetch patents
    patents = await acq.fetch_patents(query="query", limit=100)
    patents = await acq.fetch_patents(ipc_codes=["A61K"], limit=100)
    
    # Fetch chemicals
    compounds = await acq.fetch_compounds("aspirin", limit=50)
    smiles = await acq.get_smiles("ibuprofen")
    
    # Fetch sequences
    proteins = await acq.fetch_proteins("PD-1 human", limit=50)
    sequence = await acq.get_sequence("P01308")
    
    # Fetch patent sequences (NCBI)
    patent_seqs = await acq.fetch_patent_sequences("antibody", limit=50)
    
    # Fetch patent chemicals (SureChEMBL)
    patent_chems = await acq.fetch_patent_chemicals("US10000001")
    
    # Build corpus
    corpus = acq.build_corpus(articles + patents)
    chemicals = acq.extract_chemicals(patents)
    sequences = acq.extract_sequences(proteins)
```

### Document

Standard document representation.

```python
from biopat.data import Document

doc = Document(
    id="pubmed:12345678",
    source="pubmed",
    title="Article Title",
    text="Abstract text...",
    url="https://pubmed.ncbi.nlm.nih.gov/12345678",
    date="2024-01-15",
    smiles="CC(=O)O",           # Optional
    sequence="MVLSPADKTNV...",  # Optional
    sequence_type="protein",    # Optional
    metadata={"pmid": "12345678"},
)

# Convert to dict
doc_dict = doc.to_dict()

# Convert to corpus entry
corpus_entry = doc.to_corpus_entry()
```

## Retrieval

### SparseRetriever (BM25)

```python
from biopat.retrieval import SparseRetriever

retriever = SparseRetriever(k1=1.5, b=0.75)

# Index corpus
retriever.index_corpus({
    "doc1": {"title": "...", "text": "..."},
    "doc2": {"title": "...", "text": "..."},
})

# Search
results = retriever.search("query text", top_k=100)
# Returns: List[Tuple[doc_id, score]]
```

### DenseRetriever

```python
from biopat.retrieval import DenseRetriever, create_domain_retriever

# Using factory function
retriever = create_domain_retriever(
    model_name="allenai/scibert_scivocab_uncased",
    device="cuda",
)

# Index corpus
retriever.index(corpus_texts, corpus_ids)

# Search
results = retriever.search(query, top_k=100)
```

### HybridRetriever

```python
from biopat.retrieval import HybridRetriever, HybridConfig, create_hybrid_retriever

# Using factory function
retriever = create_hybrid_retriever(
    dense_model="BAAI/bge-base-en-v1.5",
    fusion_method="rrf",
)

# Or with custom config
config = HybridConfig(
    fusion_method="rrf",
    sparse_weight=0.4,
    dense_weight=0.6,
    rrf_k=60,
)
retriever = HybridRetriever(config=config)

# Index and search
retriever.index(corpus)
results = retriever.search(query, top_k=100)
```

### SPLADERetriever

```python
from biopat.retrieval import SPLADERetriever, create_splade_retriever

retriever = create_splade_retriever(
    model_name="naver/splade-cocondenser-ensembledistil"
)

# Index
retriever.index(corpus_texts, corpus_ids)

# Search
results = retriever.search(query, top_k=100)

# Get expanded terms (for interpretability)
terms = retriever.get_query_terms(query, top_k=20)
```

### ColBERTRetriever

```python
from biopat.retrieval import ColBERTRetriever, create_colbert_retriever

retriever = create_colbert_retriever(
    model_name="colbert-ir/colbertv2.0"
)

# Index
retriever.index(corpus_texts, corpus_ids)

# Search with late interaction
results = retriever.search(query, top_k=100)

# Explain match
explanation = retriever.explain(query, doc_text)
```

### CrossEncoderReranker

```python
from biopat.retrieval import CrossEncoderReranker, create_reranker

reranker = create_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Rerank results
reranked = reranker.rerank(
    query="query text",
    candidates=initial_results,  # List[Tuple[doc_id, text, score]]
    top_k=20,
)
```

### HyDEQueryExpander

```python
from biopat.retrieval import HyDEQueryExpander, create_hyde_expander

expander = create_hyde_expander(
    llm_provider="openai",
    model="gpt-4",
)

# Generate hypothetical document
hypothetical = expander.expand(query)

# Use for retrieval
results = retriever.search(hypothetical, top_k=100)
```

## Learning-to-Rank

### LambdaMARTRanker

```python
from biopat.retrieval import LambdaMARTRanker, RankingFeatures, create_ltr_ranker

# Create ranker
ranker = create_ltr_ranker(method="lambdamart")

# Prepare training data
features = [
    RankingFeatures(
        query_id="Q1",
        doc_id="D1",
        bm25_score=0.8,
        dense_score=0.9,
        reranker_score=0.85,
        relevance=2,
    ),
    # ... more features
]
group_sizes = [5, 5, 5]  # docs per query

# Train
ranker.fit(features, group_sizes)

# Predict
scores = ranker.predict(test_features)

# Rerank
ranked = ranker.rerank("Q1", test_features)
```

## Biomedical NER

### BiomedicalNER

```python
from biopat.retrieval import BiomedicalNER, create_biomedical_ner

ner = create_biomedical_ner(device=-1)  # CPU

# Extract entities
entities = ner.extract_entities(
    "Pembrolizumab treats melanoma",
    entity_types=["CHEMICAL", "DISEASE"],
)

# Extract claim structure
claim = ner.extract_claim(
    "A method of treating melanoma comprising administering pembrolizumab."
)
print(claim.chemicals)  # [BioEntity(text="pembrolizumab", ...)]
print(claim.diseases)   # [BioEntity(text="melanoma", ...)]
```

## Thesaurus

### UnifiedThesaurus

```python
from biopat.retrieval import UnifiedThesaurus, create_unified_thesaurus

thesaurus = create_unified_thesaurus()

# Lookup term
term = thesaurus.lookup("pembrolizumab")
print(term.synonyms)  # ["Keytruda", "MK-3475", ...]

# Expand query
result = thesaurus.expand_query("pembrolizumab cancer")
print(result.expanded_query)
print(result.added_terms)
```

## Molecular Search

### MolecularRetriever

```python
from biopat.retrieval import MolecularRetriever, create_molecular_retriever

retriever = create_molecular_retriever(method="fingerprint")

# Index molecules
retriever.index(smiles_list, doc_ids)

# Search by similarity
results = retriever.search(query_smiles, top_k=100, threshold=0.7)
```

### SubstructureSearcher

```python
from biopat.retrieval import SubstructureSearcher, create_substructure_searcher

searcher = create_substructure_searcher()

# Check substructure
has_benzene = searcher.has_substructure("c1ccccc1", target_smiles)

# Search database
matches = searcher.search_database(
    query_smiles="c1ccccc1",
    database=[(doc_id, smiles), ...],
)
```

## Diversity Reranking

### MMRDiversifier

```python
from biopat.retrieval import MMRDiversifier, create_mmr_diversifier

diversifier = create_mmr_diversifier(lambda_param=0.7)

# Set embeddings for similarity
diversifier.set_embeddings(doc_embeddings)

# Rerank with diversity
diverse_results = diversifier.rerank(candidates, k=10)
```

## Evaluation

### MetricsComputer

```python
from biopat.evaluation.metrics import MetricsComputer

metrics = MetricsComputer()

# Compute metrics
p_at_10 = metrics.precision_at_k(retrieved, relevant, k=10)
r_at_10 = metrics.recall_at_k(retrieved, relevant, k=10)
ndcg_10 = metrics.ndcg_at_k(retrieved, relevance_scores, k=10)
mrr = metrics.mrr(retrieved, relevant)
map_score = metrics.map(retrieved, relevant)
```

## Training

### DenseRetrieverTrainer

```python
from biopat.training import DenseRetrieverTrainer, TrainingConfig, TrainingExample

config = TrainingConfig(
    model_name="allenai/scibert_scivocab_uncased",
    epochs=3,
    learning_rate=2e-5,
)

trainer = DenseRetrieverTrainer(config)

# Prepare data
examples = [
    TrainingExample(
        query="query text",
        positive="relevant doc",
        negatives=["irrelevant doc 1", "irrelevant doc 2"],
    ),
]

# Train
metrics = trainer.train(examples)

# Save
trainer.save("models/fine_tuned")
```
