# BioPAT Datasheet

Following the "Datasheets for Datasets" framework (Gebru et al., 2018).

## Motivation
- **Why was the dataset created?** To fill the gap in cross-domain retrieval between patent claims and scientific literature.
- **Who created it?** `[Your Organization/Name]`.
- **Who funded the creation?** `[PENDING]`.

## Composition
- **What do instances represent?** Each instance is a pair consisting of a specific biomedical patent claim (query) and a scientific research paper abstract (document).
- **How many instances are there?** Target: ~5,000 queries and ~100,000+ documents (Phase 1).
- **Is there a label or target?** Yes, 4-tier graded relevance scores (0-3).

## Collection Process
- **How was the data collected?** Aggregated via public APIs (PatentsView, OpenAlex) and curated bulk datasets (Reliance on Science, USPTO Office Actions).
- **Who was involved?** Automated pipeline with periodic manual quality audits.

## Preprocessing/Cleaning/Labeling
- **Was any preprocessing done?** Claims were parsed for independent status; abstracts were reconstructed from inverted indices; temporal constraints were applied to ensure papers predate patent priority.

## Distribution
- **How is it distributed?** Available via GitHub (code/metadata) and Zenodo (bulk data).
- **License?** CC-BY-NC-SA 4.0.
