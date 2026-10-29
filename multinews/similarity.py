import numpy as np

def tfidf_similarity(texts, query=None):
    """
    Compute TF-IDF cosine similarity matrix and optional query relevance.

    No heavy deps — uses only stdlib + numpy.

    Args:
        texts: list of passage strings
        query: optional query string; if provided, also returns
               per-passage relevance weights (cosine with query)

    Returns:
        sim_matrix: (n, n) similarity matrix
        If query is provided, returns (sim_matrix, query_weights)
        where query_weights[i] = cosine(passage_i, query), clipped to [0, ∞).
    """
    import re
    from collections import Counter

    # Tokenize passages (and query if given)
    all_texts = list(texts) + ([query] if query else [])
    docs = []
    for t in all_texts:
        tokens = re.findall(r"\w+", t.lower())
        docs.append(Counter(tokens))

    # Build vocabulary
    vocab = {}
    for doc in docs:
        for w in doc:
            if w not in vocab:
                vocab[w] = len(vocab)

    n_passages = len(texts)
    n_total = len(docs)
    V = len(vocab)

    # TF-IDF matrix
    df = np.zeros(V)
    for doc in docs:
        for w in doc:
            df[vocab[w]] += 1
    idf = np.log((n_total + 1) / (df + 1)) + 1  # smooth IDF

    mat = np.zeros((n_total, V))
    for i, doc in enumerate(docs):
        for w, cnt in doc.items():
            mat[i, vocab[w]] = cnt * idf[vocab[w]]

    # Normalize rows
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    mat = mat / norms

    # Passage-passage similarity
    passage_mat = mat[:n_passages]
    sim_matrix = passage_mat @ passage_mat.T

    if query is not None:
        query_vec = mat[n_passages]  # last row
        query_weights = passage_mat @ query_vec
        query_weights = np.maximum(query_weights, 0.0)  # clip negatives
        return sim_matrix, query_weights

    return sim_matrix
