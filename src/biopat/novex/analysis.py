"""NovEx Analysis — vocabulary gap, per-domain, cross-domain, agreement, paper tables."""

import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from biopat.novex.benchmark import NovExBenchmark
from biopat.novex.evaluator import TierResult

logger = logging.getLogger(__name__)

STOPWORDS = frozenset(
    "the a an is are was were be been being have has had do does did will would could "
    "should may might shall can to of in for on with at by from as into through during "
    "before after between out off over under again then once here there when where why "
    "how all both each few more most other some such no nor not only own same so than "
    "too very and but if or because until while that which this these those it its we "
    "our they their".split()
)


def _tokenize(text: str) -> set:
    return set(text.lower().split()) - STOPWORDS


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x)/n, sum(y)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a-mx)**2 for a in x))
    sy = math.sqrt(sum((b-my)**2 for b in y))
    return cov / (sx * sy) if sx and sy else 0.0


class NovExAnalyzer:
    """Generate all analysis outputs for the NovEx paper."""

    def __init__(self, benchmark: NovExBenchmark, results: List[TierResult],
                 output_dir: str = "data/novex/analysis"):
        self.b = benchmark
        self.results = results
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.t1 = [r for r in results if r.tier == 1]
        self.t2 = [r for r in results if r.tier == 2]
        self.t3 = [r for r in results if r.tier == 3]

    def run_all(self) -> Dict[str, Any]:
        out = {
            "vocabulary_gap": self.vocabulary_gap(),
            "per_domain": self.per_domain(),
            "cross_domain": self.cross_domain(),
            "doc_type_split": self.doc_type_split(),
            "tier1_table": self.tier1_table(),
            "tier2_table": self.tier2_table(),
            "tier3_table": self.tier3_table(),
            "summary": self.summary(),
        }
        self._save("full_analysis.json", out)
        logger.info("Analysis saved to %s", self.out)
        return out

    def vocabulary_gap(self) -> Dict:
        rows = []
        bm25 = next((r for r in self.t1 if r.method == "bm25"), None)
        for sid, s in self.b.statements.items():
            sw = _tokenize(s.text)
            rel = self.b.tier1_qrels.get(sid, {})
            if not sw or not rel:
                continue
            overlaps = []
            for did in rel:
                doc = self.b.corpus.get(did, {})
                dw = _tokenize(doc.get("title", "") + " " + doc.get("text", ""))
                if dw:
                    overlaps.append(len(sw & dw) / len(sw | dw))
            recall = bm25.per_query.get(sid, {}).get("recall@100", 0) if bm25 else 0
            rows.append({"id": sid, "domain": s.domain, "overlap": _mean(overlaps), "bm25_r100": recall})
        corr = _pearson([r["overlap"] for r in rows], [r["bm25_r100"] for r in rows])
        out = {"per_statement": rows, "correlation": corr}
        self._save("vocabulary_gap.json", out)
        return out

    def per_domain(self) -> Dict:
        domains = sorted(set(s.domain[:3] for s in self.b.statements.values()))
        out = {}
        for d in domains:
            ids = {sid for sid, s in self.b.statements.items() if s.domain.startswith(d)}
            dm = {}
            for r in self.t1:
                vals = [m for qid, m in r.per_query.items() if qid in ids]
                if vals:
                    dm[f"{r.method}/{r.model}"] = {k: _mean([v[k] for v in vals]) for k in vals[0]}
            out[d] = {"count": len(ids), "metrics": dm}
        self._save("per_domain.json", out)
        return out

    def cross_domain(self) -> Dict:
        in_cats = {"multi_patent_examiner", "single_patent_examiner"}
        out = {}
        for r in self.t1:
            in_r = [m.get("recall@100", 0) for qid, m in r.per_query.items()
                    if self.b.statements.get(qid) and self.b.statements[qid].category in in_cats]
            cx_r = [m.get("recall@100", 0) for qid, m in r.per_query.items()
                    if self.b.statements.get(qid) and self.b.statements[qid].category == "cross_domain"]
            if in_r and cx_r:
                out[f"{r.method}/{r.model}"] = {"in_domain": _mean(in_r), "cross_domain": _mean(cx_r),
                                                 "gap": _mean(in_r) - _mean(cx_r)}
        self._save("cross_domain.json", out)
        return out

    def doc_type_split(self) -> Dict:
        out = {f"{r.method}/{r.model}": {k: v for k, v in r.metrics.items()
               if "paper_recall" in k or "patent_recall" in k or "recall@" in k}
               for r in self.t1}
        self._save("doc_type_split.json", out)
        return out

    def tier1_table(self) -> List[Dict]:
        rows = [{"method": r.method, "model": r.model, **{k: round(v, 4) for k, v in r.metrics.items()}} for r in self.t1]
        self._save("tier1_table.json", rows)
        self._latex("tier1.tex", ["Method", "Model", "R@10", "R@50", "R@100", "NDCG@10", "MAP"],
                    [[r["method"], r["model"]] + [f'{r.get(k,0):.3f}' for k in ["recall@10","recall@50","recall@100","ndcg@10","map"]] for r in rows])
        return rows

    def tier2_table(self) -> List[Dict]:
        rows = [{"model": r.model, **{k: round(v, 4) for k, v in r.metrics.items()}} for r in self.t2]
        self._save("tier2_table.json", rows)
        return rows

    def tier3_table(self) -> List[Dict]:
        rows = [{"model": r.model, "context": r.metadata.get("with_context", True),
                 **{k: round(v, 4) for k, v in r.metrics.items()}} for r in self.t3]
        self._save("tier3_table.json", rows)
        return rows

    def summary(self) -> Dict:
        best_t1 = max(self.t1, key=lambda r: r.metrics.get("recall@100", 0)) if self.t1 else None
        best_t2 = max(self.t2, key=lambda r: r.metrics.get("weighted_kappa", 0)) if self.t2 else None
        best_t3 = max(self.t3, key=lambda r: r.metrics.get("macro_f1", 0)) if self.t3 else None
        out = {
            "benchmark": self.b.get_stats(),
            "best_t1": {"method": best_t1.method, "recall@100": best_t1.metrics.get("recall@100")} if best_t1 else None,
            "best_t2": {"model": best_t2.model, "kappa": best_t2.metrics.get("weighted_kappa")} if best_t2 else None,
            "best_t3": {"model": best_t3.model, "f1": best_t3.metrics.get("macro_f1")} if best_t3 else None,
            "total_cost": sum(r.cost_usd for r in self.results),
        }
        self._save("summary.json", out)
        return out

    def _save(self, name: str, data):
        with open(self.out / name, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _latex(self, name: str, headers: List[str], rows: List[List[str]]):
        lines = ["\\begin{table}[t]", "\\centering",
                 f"\\begin{{tabular}}{{{'l' + 'c' * (len(headers)-1)}}}",
                 "\\toprule",
                 " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
                 "\\midrule"]
        for row in rows:
            lines.append(" & ".join(row) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        with open(self.out / name, "w") as f:
            f.write("\n".join(lines))
