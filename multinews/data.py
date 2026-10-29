import re
import numpy as np
import json
from pathlib import Path

def chunk_text(text, target_words=150):
    """
    Split text into ~target_words chunks at sentence boundaries.

    Uses regex sentence splitting (handles Mr./Dr./etc. abbreviations).
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current_words + sent_words > target_words * 1.3 and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_words = sent_words
        else:
            current.append(sent)
            current_words += sent_words

    if current:
        chunks.append(" ".join(current))

    # Merge very short trailing chunks
    if len(chunks) > 1 and len(chunks[-1].split()) < target_words * 0.3:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()

    return chunks


def build_example(example, all_examples, n_target, chunk_words=150,
                  seed=42):
    """
    Build a passage pool for one Multi-News example.

    1. Chunk source articles into ~chunk_words passages (core passages)
    2. Add distractor passages from other clusters to reach n_target

    Args:
        example: dict with 'articles', 'summary', 'idx'
        all_examples: list of all examples (for distractors)
        n_target: target pool size
        chunk_words: target words per chunk
        seed: random seed for distractor sampling

    Returns dict with:
        passages: list of passage strings
        is_source: list of bools (True = from source articles)
        source_article_idx: list of ints (-1 for distractors)
        summary: reference summary
        query: constructed query (titles/first sentences)
        n_source: number of source passages
    """
    rng = np.random.RandomState(seed + example["idx"])

    # Chunk source articles
    source_passages = []
    source_article_ids = []
    for art_idx, article in enumerate(example["articles"]):
        chunks = chunk_text(article, target_words=chunk_words)
        for chunk in chunks:
            source_passages.append(chunk)
            source_article_ids.append(art_idx)

    n_source = len(source_passages)

    # Build query from first sentences of each article
    query_parts = []
    for article in example["articles"]:
        first_sent = re.split(r'(?<=[.!?])\s+', article.strip())
        if first_sent:
            query_parts.append(first_sent[0][:200])
    query = " ".join(query_parts)

    # Add distractors if needed
    n_distractors = max(0, n_target - n_source)
    distractor_passages = []

    if n_distractors > 0:
        # Pool chunks from other clusters
        other_chunks = []
        for other in all_examples:
            if other["idx"] == example["idx"]:
                continue
            for article in other["articles"]:
                chunks = chunk_text(article, target_words=chunk_words)
                other_chunks.extend(chunks)

        # Sample distractors
        if len(other_chunks) >= n_distractors:
            chosen = rng.choice(len(other_chunks), size=n_distractors,
                                replace=False)
            distractor_passages = [other_chunks[i] for i in chosen]
        else:
            distractor_passages = other_chunks

    # Combine and shuffle
    passages = source_passages + distractor_passages
    is_source = [True] * n_source + [False] * len(distractor_passages)
    article_ids = source_article_ids + [-1] * len(distractor_passages)

    # Shuffle (preserving alignment)
    perm = rng.permutation(len(passages))
    passages = [passages[i] for i in perm]
    is_source = [is_source[i] for i in perm]
    article_ids = [article_ids[i] for i in perm]

    return {
        "passages": passages,
        "is_source": is_source,
        "source_article_idx": article_ids,
        "summary": example["summary"],
        "query": query,
        "n_source": n_source,
        "example_idx": example["idx"],
    }


from release_paths import DATASET
def load_multinews(max_examples=300, seed=42):
    data=json.loads(DATASET.read_text())
    rng=np.random.RandomState(seed)
    indices=rng.choice(len(data),size=min(max_examples,len(data)),replace=False)
    return [data[i] for i in sorted(indices)]
