"""Baseline 3: Embedding Similarity (simulates RAG). Top-k by TF-IDF cosine similarity."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cognitive_cache.models import Source, Task, ScoredSource, SelectionResult
from cognitive_cache.baselines.base import BaselineStrategy


class EmbeddingStrategy(BaselineStrategy):
    def select(self, sources: list[Source], task: Task, budget: int) -> SelectionResult:
        if not sources:
            return SelectionResult(selected=[], total_tokens=0, budget=budget)

        corpus = [task.full_text] + [s.content for s in sources]
        try:
            vectorizer = TfidfVectorizer(max_features=5000, token_pattern=r"(?u)\b\w+\b")
            matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            return SelectionResult(selected=[], total_tokens=0, budget=budget)

        task_vec = matrix[0:1]
        source_vecs = matrix[1:]
        similarities = cosine_similarity(task_vec, source_vecs)[0]

        ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

        selected = []
        total = 0
        for idx, sim in ranked:
            s = sources[idx]
            if total + s.token_count > budget:
                continue
            selected.append(ScoredSource(source=s, score=float(sim)))
            total += s.token_count

        return SelectionResult(selected=selected, total_tokens=total, budget=budget)
