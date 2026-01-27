# Methodology

This document outlines the detailed construction methodology for the BioPAT (Biomedical Patent-to-Article Retrieval) benchmark.

## 3.1 Patent Selection Criteria

Patents were selected based on the following criteria:
1. **Domain**: IPC codes beginning with `A61` (Medical/Veterinary), `C07` (Organic Chemistry), or `C12` (Biochemistry).
2. **Evidence**: At least 3 Non-Patent Literature (NPL) citations in the Reliance on Science dataset with a confidence score ≥ 8.
3. **Temporal Window**: Filing date between 2008-2020 to ensure both modern patenting practices and sufficient literature coverage.

This Yielded N=`[X]` candidate patents, from which we sampled...

## 3.2 Ground Truth Construction

Relevance was assigned using a 4-tier graded scale (0-3) to reflect the nuance of prior art search:

- **Score 3 (Novelty-Destroying)**: A specific reference cited in a USPTO §102 rejection within an Office Action. The reference is determined by an examiner to anticipate the claim.
- **Score 2 (Highly Relevant)**: A reference cited in a USPTO §103 rejection (obviousness) OR an examiner citation with a Reliance on Science confidence score ≥ 9.
- **Score 1 (Relevant)**: An examiner citation with confidence 7-8 OR an applicant-provided citation with confidence ≥ 8. These serve as useful background or state-of-the-art context.
- **Score 0 (Irrelevant)**: Default for unlinked documents OR any document published *after* the patent priority date (Hard Temporal Constraint).

## 3.3 Evaluation Metrics

The benchmark is evaluated using standard IR metrics, emphasizing top-heavy ranking:
- **NDCG@10**: Primary metric for graded relevance.
- **Recall@100**: Critical for measuring the comprehensiveness required in legal prior art searches.
- **MRR**: Mean Reciprocal Rank.

## 3.4 Data Access Strategy

**Primary approach:** API-based retrieval for all sources.

| Data Source | Method | Rationale |
|-------------|--------|-----------|
| PatentsView | API | Filtering, lower storage |
| Reliance on Science | Bulk (Zenodo) | Pre-computed, small file |
| Office Actions | Bulk (USPTO) | No API available |
| OpenAlex | API | No approval needed |
| NCBI | API | MeSH enrichment |

Bulk downloads (PatentsView, USPTO XML) remain available as fallback if API rate limits cause issues.
