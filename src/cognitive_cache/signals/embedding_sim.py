"""Signal 5: Embedding Similarity (semantic fallback).

Uses TF-IDF vectors to compute similarity between the task text and file content.
This is the same basic idea as RAG (vector similarity search), but we weight it
LOW in the final score because we're trying to beat RAG, not replicate it.

Why TF-IDF instead of neural embeddings: it's fast, free (no API calls),
needs no GPU, and for code (which has distinctive vocabulary) it works
surprisingly well. We can upgrade to neural embeddings later if needed.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cognitive_cache.models import Source, Task
from cognitive_cache.signals.base import Signal


class EmbeddingSimilaritySignal(Signal):
    def __init__(self):
        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            token_pattern=r"(?u)\b\w+\b",  # match single-char tokens too
            stop_words="english",
        )
        self._fitted = False
        self._corpus_paths: list[str] = []
        self._corpus_matrix = None

    def fit(self, sources: list[Source]):
        """Pre-compute TF-IDF vectors for all source files.

        Call this once before scoring. It builds the vocabulary
        from all files so that similarity scores are meaningful.
        """
        corpus = [s.content for s in sources]
        self._corpus_paths = [s.path for s in sources]
        if not corpus:
            return
        self._corpus_matrix = self._vectorizer.fit_transform(corpus)
        self._fitted = True

    def score(self, source: Source, task: Task, selected: list[Source], **kwargs) -> float:
        if not self._fitted or self._corpus_matrix is None:
            # Fallback: fit on just this source and the task
            try:
                vectorizer = TfidfVectorizer(max_features=5000, token_pattern=r"(?u)\b\w+\b")
                matrix = vectorizer.fit_transform([task.full_text, source.content])
                sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
                return float(max(0.0, sim))
            except ValueError:
                return 0.0

        # Find this source in the pre-computed matrix
        try:
            idx = self._corpus_paths.index(source.path)
        except ValueError:
            return 0.0

        # Compute similarity between task and this source
        task_vec = self._vectorizer.transform([task.full_text])
        sim = cosine_similarity(task_vec, self._corpus_matrix[idx:idx + 1])[0][0]
        return float(max(0.0, sim))
