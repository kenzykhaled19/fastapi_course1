"""
search.py — Member 3: Search & Ranking
Handles: spelling correction, wildcard queries, TF-IDF scoring, cosine similarity ranking
Input:  inverted_index.json, corpus.json, doc_id_map.json
Output: top 5 ranked documents for a user query
"""

import json
import math
import re
import os
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# ─── Download required NLTK data (only runs once) ────────────────────────────
nltk.download('stopwords', quiet=True)

# ─── Paths — adjust if your files are in a different folder ──────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INVERTED_INDEX_PATH = os.path.join(BASE_DIR, "inverted_index.json")
CORPUS_PATH         = os.path.join(BASE_DIR, "corpus.json")
DOC_ID_MAP_PATH     = os.path.join(BASE_DIR, "doc_id_map.json")

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading index and corpus... (this takes a few seconds)")

with open(INVERTED_INDEX_PATH, "r", encoding="utf-8") as f:
    inverted_index = json.load(f)

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

with open(DOC_ID_MAP_PATH, "r", encoding="utf-8") as f:
    doc_id_map = json.load(f)  # {"1": "filename.txt", ...}

print(f"Index loaded: {len(inverted_index)} unique terms")
print(f"Corpus loaded: {len(corpus)} documents\n")

# ─── Setup ────────────────────────────────────────────────────────────────────
stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))
N          = len(doc_id_map)  # total number of documents = 92

# Vocabulary = all stemmed terms in the index
VOCABULARY = list(inverted_index.keys())

# Pre-build: reverse map from filename → doc_id (integer)
filename_to_id = {v: int(k) for k, v in doc_id_map.items()}

# Pre-build: for each doc_id, a Counter of its stemmed tokens (for TF)
# corpus.json keys are document names without .txt — we map them to doc_ids
print("Building term frequency table...")
doc_tf = {}  # doc_tf[doc_id] = Counter({term: count, ...})

for doc_name, doc_data in corpus.items():
    filename = doc_data["filename"]
    if filename in filename_to_id:
        doc_id = filename_to_id[filename]
        tokens = doc_data["processed_tokens"]
        doc_tf[doc_id] = Counter(tokens)
    
print("Ready.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. QUERY PREPROCESSING
#    Same pipeline as Member 1: lowercase → tokenize → remove stopwords → stem
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_query(query: str) -> list[str]:
    """Clean and stem the user's query the same way Member 1 processed documents."""
    query = query.lower()
    # Split on anything that isn't a letter, digit, or wildcard *
    tokens = re.findall(r'[a-z0-9\*]+', query)
    result = []
    for token in tokens:
        if '*' in token:
            result.append(token)          # wildcard — don't stem, handle separately
        elif token not in stop_words:
            result.append(stemmer.stem(token))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SPELLING CORRECTION — Edit Distance
#    If a query term isn't in the index, find the closest vocabulary word.
#
#    Edit distance = minimum number of single-character edits (insert, delete,
#    replace) to turn one word into another.
#    e.g. "bactria" → "bacteria" costs 1 insertion → distance = 1
# ═══════════════════════════════════════════════════════════════════════════════

def edit_distance(s1: str, s2: str) -> int:
    """Classic dynamic programming edit distance (Levenshtein)."""
    m, n = len(s1), len(s2)
    # dp[i][j] = edit distance between s1[:i] and s2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i        # deleting i chars from s1
    for j in range(n + 1):
        dp[0][j] = j        # inserting j chars into s1

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]          # characters match, no cost
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # replacement
                )
    return dp[m][n]


def spell_correct(term: str, max_distance: int = 2) -> str | None:
    """
    If term is not in the index, find the closest vocabulary word within
    max_distance edits. Returns the corrected term or None if nothing close enough.
    """
    if term in inverted_index:
        return term  # already correct, no fix needed

    best_term     = None
    best_distance = max_distance + 1  # start worse than threshold

    for vocab_term in VOCABULARY:
        # Small optimization: skip if length difference alone exceeds threshold
        if abs(len(vocab_term) - len(term)) > max_distance:
            continue
        dist = edit_distance(term, vocab_term)
        if dist < best_distance:
            best_distance = dist
            best_term     = vocab_term

    return best_term  # None if nothing within max_distance


# ═══════════════════════════════════════════════════════════════════════════════
# 3. WILDCARD QUERY SUPPORT
#    User types "gram*" → we find all index terms that start with "gram"
#    User types "*coli"  → all terms ending with "coli"
#    User types "e*oli"  → all terms matching that pattern
#
#    Since we don't have a permuterm index, we do a vocabulary scan.
#    Fine for 92 documents.
# ═══════════════════════════════════════════════════════════════════════════════

def expand_wildcard(pattern: str) -> list[str]:
    """
    Convert a wildcard pattern like 'gram*' into a list of matching index terms.
    The * can appear anywhere in the pattern.
    """
    # Convert wildcard pattern to a regex: * → .*
    regex_pattern = re.escape(pattern).replace(r'\*', '.*')
    regex = re.compile(f'^{regex_pattern}$')
    matches = [term for term in VOCABULARY if regex.match(term)]
    return matches


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TF-IDF SCORING + COSINE SIMILARITY RANKING
#
#    TF  (Term Frequency)  = how often does this term appear in this document?
#         tf(t, d) = count(t in d) / total_tokens(d)
#         → normalized so long documents don't automatically win
#
#    IDF (Inverse Document Frequency) = how rare is this term across all docs?
#         idf(t) = log( N / df(t) )
#         where df(t) = number of documents containing term t
#         → rare terms get higher weight; "bacteria" appearing in all 92 docs
#           is less useful than "carbapenem" appearing in only 3
#
#    TF-IDF(t, d) = tf(t, d) × idf(t)
#
#    Cosine Similarity = how similar is the query vector to a document vector?
#         score(q, d) = Σ [tfidf(t,q) × tfidf(t,d)] / (||q|| × ||d||)
#         → normalizes for document length so a short precise article can
#           beat a long rambling one
# ═══════════════════════════════════════════════════════════════════════════════

def compute_idf(term: str) -> float:
    """IDF = log(N / number_of_docs_containing_term)"""
    if term not in inverted_index:
        return 0.0
    df = len(inverted_index[term]["postings"])
    return math.log(N / df) if df > 0 else 0.0


def compute_tf(term: str, doc_id: int) -> float:
    """TF = count of term in doc / total tokens in doc"""
    if doc_id not in doc_tf:
        return 0.0
    token_counts = doc_tf[doc_id]
    total_tokens = sum(token_counts.values())
    return token_counts.get(term, 0) / total_tokens if total_tokens > 0 else 0.0


def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Full search pipeline:
      1. Preprocess query
      2. Correct typos
      3. Expand wildcards
      4. Score all candidate documents with TF-IDF + cosine similarity
      5. Return top_k results
    """

    # Step 1: preprocess
    query_terms = preprocess_query(query)
    if not query_terms:
        print("Query is empty after preprocessing.")
        return []

    # Step 2 & 3: correct and expand each term
    final_terms   = []   # the actual index terms we'll search
    corrections   = {}   # track what was corrected for user feedback

    for term in query_terms:
        if '*' in term:
            # Wildcard — expand, no stemming/correction needed
            expanded = expand_wildcard(term)
            if expanded:
                print(f"  Wildcard '{term}' expanded to: {expanded}")
                final_terms.extend(expanded)
            else:
                print(f"  Wildcard '{term}' matched nothing.")
        else:
            if term in inverted_index:
                final_terms.append(term)
            else:
                corrected = spell_correct(term)
                if corrected:
                    corrections[term] = corrected
                    final_terms.append(corrected)
                else:
                    print(f"  '{term}' not found and no close match — skipping.")

    # Report spelling corrections to user
    if corrections:
        for original, fixed in corrections.items():
            print(f"  Did you mean '{fixed}' instead of '{original}'?")

    if not final_terms:
        print("No valid search terms found.")
        return []

    # Remove duplicates but keep order
    seen = set()
    unique_terms = [t for t in final_terms if not (t in seen or seen.add(t))]

    # Step 4: find candidate documents and score them

    # Gather all candidate doc IDs (union of postings for all query terms)
    candidate_docs = set()
    for term in unique_terms:
        if term in inverted_index:
            candidate_docs.update(inverted_index[term]["postings"])

    if not candidate_docs:
        print("No documents found for this query.")
        return []

    # Compute query vector (TF-IDF weights for query terms)
    # For the query itself, TF = 1/len(query_terms) for each term (uniform)
    query_tfidf = {}
    for term in unique_terms:
        idf = compute_idf(term)
        query_tfidf[term] = (1 / len(unique_terms)) * idf

    query_norm = math.sqrt(sum(v ** 2 for v in query_tfidf.values()))

    # Compute document scores
    scores = {}
    for doc_id in candidate_docs:
        dot_product = 0.0
        doc_norm_sq = 0.0

        for term in unique_terms:
            tf  = compute_tf(term, doc_id)
            idf = compute_idf(term)
            doc_weight   = tf * idf
            query_weight = query_tfidf.get(term, 0.0)

            dot_product += query_weight * doc_weight
            doc_norm_sq += doc_weight ** 2

        doc_norm = math.sqrt(doc_norm_sq)
        denominator = query_norm * doc_norm

        if query_norm > 0:
            scores[doc_id] = dot_product / query_norm
        else:
            scores[doc_id] = 0.0

    # Step 5: sort and return top_k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for doc_id, score in ranked:
        filename = doc_id_map.get(str(doc_id), f"doc_{doc_id}")
        results.append({
            "rank":     len(results) + 1,
            "doc_id":   doc_id,
            "filename": filename,
            "score":    round(score, 4)
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN — Interactive search loop
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(results: list[dict]):
    if not results:
        print("No results found.\n")
        return
    print(f"\n{'─'*55}")
    print(f"  Top {len(results)} Results")
    print(f"{'─'*55}")
    for r in results:
        print(f"  #{r['rank']}  [score: {r['score']:.4f}]  {r['filename']}")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    print("=" * 55)
    print("  Waterborne Bacteria Search Engine")
    print("  Type 'quit' to exit")
    print("  Supports wildcards: gram*   *coli   anti*tic")
    print("=" * 55 + "\n")

    while True:
        query = input("Search > ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        print()
        results = search(query, top_k=5)
        print_results(results)