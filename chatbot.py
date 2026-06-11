import os, re, numpy as np
from collections import Counter
from groq import Groq
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

CHATBOT_NAME = "HydroBot"
CHATBOT_ROLE = "a medical assistant specialized in waterborne bacteria, infections, and treatments"

# ── Thesaurus ──
THESAURUS = {
    "e coli":        ["escherichia coli", "coliform", "gram negative rod"],
    "ecoli":         ["escherichia coli", "coliform", "e coli"],
    "salmonella":    ["salmonellosis", "typhoid", "gram negative enterobacteria"],
    "cholera":       ["vibrio cholerae", "waterborne disease", "gram negative"],
    "legionella":    ["legionnaires disease", "waterborne pathogen", "gram negative"],
    "pseudomonas":   ["gram negative rod", "aerobic bacteria", "opportunistic pathogen"],
    "staph":         ["staphylococcus", "gram positive cocci", "s aureus", "mrsa"],
    "staphylococcus":["staph", "gram positive cocci", "s aureus", "mrsa"],
    "strep":         ["streptococcus", "gram positive cocci"],
    "mrsa":          ["methicillin resistant staphylococcus", "antibiotic resistant staph"],
    "gram negative": ["gram-negative", "gnb", "enterobacteriaceae"],
    "gram positive": ["gram-positive", "gpb", "firmicutes"],
    "treatment":     ["therapy", "antibiotic", "antimicrobial", "medication"],
    "antibiotic":    ["antimicrobial", "antibacterial", "drug"],
    "infection":     ["contamination", "disease", "pathogen"],
    "water":         ["aquatic", "waterborne", "drinking water", "contamination"],
    "resistant":     ["antibiotic resistance", "drug resistant", "multidrug", "alternative treatment"],
    "resistance":    ["resistant bacteria", "multidrug resistant", "alternative antibiotic"],
    "vaccine":       ["vaccination", "immunization", "oral vaccine", "prevention"],
    "vaccines":      ["vaccination", "immunization", "cholera vaccine", "prevention"],
}

def expand_query(query):
    query_lower = query.lower()
    expanded_terms = [query_lower]
    for key, synonyms in THESAURUS.items():
        if key in query_lower:
            expanded_terms.extend(synonyms)
    seen, unique = set(), []
    for term in expanded_terms:
        if term not in seen:
            seen.add(term)
            unique.append(term)
    return " ".join(unique)

# ── Word2Vec ──
USELESS = {
    'are','and','the','was','has','been','have','that','this','with','from',
    'they','were','also','found','more','than','not','for','but','its','can',
    'may','such','some','most','inhabitants','habitats','copepods','regions',
    'hours','cells','your','symptoms','sources','rates','levels','types','cases',
    'yourself','answers','greatly','provider','worries','let','licensed',
    'formalin','rctb','bacilli','swimming','while','rational','restrict',
    'guiding','practiced','passive','transitional'
}

def load_documents(docs_folder):
    docs = []
    for fname in sorted(os.listdir(docs_folder)):
        if fname.endswith(".txt"):
            path = os.path.join(docs_folder, fname)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                docs.append((fname, f.read()))
    return docs

def train_word2vec(docs_folder):
    documents = load_documents(docs_folder)
    tokenized = []
    for _, text in documents:
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        tokenized.append(words)
    model = Word2Vec(
        sentences=tokenized, vector_size=100, window=5,
        min_count=1, sg=1, epochs=30, workers=2
    )
    print(f"[Word2Vec] Trained on {len(tokenized)} docs, vocab: {len(model.wv)} words.")
    return model

def get_word2vec_expansions(query, w2v_model, top_n=3, threshold=0.85):
    words = re.findall(r'\b[a-z]{3,}\b', query.lower())
    extra_words = []
    for word in words:
        if word in w2v_model.wv:
            similar = w2v_model.wv.most_similar(word, topn=top_n)
            for sim_word, score in similar:
                if score >= threshold and sim_word not in query.lower() and sim_word not in USELESS:
                    extra_words.append(sim_word)
    return list(dict.fromkeys(extra_words))

# ── Pseudo Relevance Feedback ──
STOPWORDS = {
    'that','this','with','from','they','were','have','been','their','also',
    'which','when','some','more','will','used','such','most','than','into',
    'other','these','those','both','about','after','before','between','your',
    'symptoms','hours','vibrio','cholerae','cholera','salmonella','pseudomonas',
    'legionella','staphylococcus','streptococcus','enterococcus','klebsiella',
    'while','cause','swimming'
}

def pseudo_relevance_feedback(original_query, top_docs_content, top_k_words=3):
    if not top_docs_content:
        return original_query
    combined = " ".join(top_docs_content[:2]).lower()
    words = re.findall(r'\b[a-z]{5,}\b', combined)
    words = [w for w in words if w not in STOPWORDS]
    freq = Counter(words)
    query_words = set(original_query.lower().split())
    new_words = []
    for word, _ in freq.most_common(20):
        if word not in query_words and len(new_words) < top_k_words:
            new_words.append(word)
    return original_query + " " + " ".join(new_words)

# ── Semantic Search ──
print("Loading semantic model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_index = []

def build_search_index(docs_folder):
    global doc_index
    doc_index = []
    for fname in os.listdir(docs_folder):
        if fname.endswith('.txt'):
            path = os.path.join(docs_folder, fname)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            chunks, start = [], 0
            while start < len(content):
                chunk = content[start:start + 500]
                if len(chunk.strip()) > 50:
                    emb = embedding_model.encode(chunk, convert_to_numpy=True)
                    doc_index.append((fname, chunk, emb))
                start += 400
    print(f"[Index] Built {len(doc_index)} chunks from {docs_folder}")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query, top_k=5):
    query_emb = embedding_model.encode(query, convert_to_numpy=True)
    scored = [
        (fname, chunk, float(cosine_similarity(query_emb, emb)))
        for fname, chunk, emb in doc_index
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    seen, results = set(), []
    for fname, chunk, score in scored:
        if fname not in seen:
            seen.add(fname)
            results.append((fname, chunk, score))
        if len(results) >= top_k:
            break
    return results

# ── Groq RAG ──
conversation_history = []

def clean_markdown(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*', '', text)
    return text.strip()

def get_confidence_label(top_score):
    if top_score >= 0.75:
        return "High confidence"
    elif top_score >= 0.60:
        return "Medium confidence"
    else:
        return "Low confidence"

def ask_groq(user_question, relevant_docs, api_key):
    global conversation_history
    context = "\n\n---\n\n".join(relevant_docs[:3])
    system_message = {
        "role": "system",
        "content": f"""You are {CHATBOT_NAME}, {CHATBOT_ROLE}.
RULES:
1. Use the knowledge base below as your primary source.
2. Answer naturally and directly. Do NOT say "the documents say".
3. If the topic is outside your knowledge base, say: "I don't have enough information on this. Please consult a specialist."
4. Keep answers clear and under 300 words.
5. Do NOT use markdown symbols like * or ** anywhere in your answer.
6. Write like a knowledgeable doctor explaining to a colleague.
   - Start with one clear sentence that directly answers the question.
   - Then explain with 2-3 sentences of context or detail.
   - If there are multiple steps or options, list them as:
     1. First point
     2. Second point
     3. Third point
   - End with one sentence on what to watch for or when to escalate.
   - Never use labels like "Key facts:" or "Treatment steps:" or "Important note:".
   - Never start a sentence with "It's worth noting" or "It's important to note".

KNOWLEDGE BASE:

{context}"""
    }
    conversation_history.append({"role": "user", "content": user_question})
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[system_message] + conversation_history,
        temperature=0.1,
        max_tokens=500
    )
    answer = clean_markdown(response.choices[0].message.content.strip())
    conversation_history.append({"role": "assistant", "content": answer})
    return answer

def reset_conversation():
    global conversation_history
    conversation_history = []

def load_conversation_history(messages):
    """
    Restores a previous conversation from the database.
    messages format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    global conversation_history
    conversation_history = messages[-10:]    

# ── Main ──
BACTERIA_LIST = [
    'e coli','ecoli','salmonella','cholera','vibrio','pseudomonas',
    'legionella','klebsiella','staph','staphylococcus','strep',
    'streptococcus','mrsa','shigella','campylobacter','listeria','enterobacter'
]

def get_answer(user_question, search_function, docs_folder, groq_api_key, w2v_model=None):
    question_lower = user_question.lower()
    detected = [b for b in BACTERIA_LIST if b in question_lower]
    if detected:
        search_query = user_question
    else:
        topic_question = None
        for msg in conversation_history:
            if msg["role"] == "user":
                if any(b in msg["content"].lower() for b in BACTERIA_LIST):
                    topic_question = msg["content"]
                    break
        if topic_question:
            search_query = topic_question + " " + user_question
        else:
            user_messages = [m for m in conversation_history if m["role"] == "user"]
            search_query = (user_messages[-1]["content"] + " " + user_question) if user_messages else user_question

    expanded = expand_query(search_query)
    if w2v_model is not None:
        w2v_words = get_word2vec_expansions(expanded, w2v_model)
        if w2v_words:
            expanded = expanded + " " + " ".join(w2v_words)

    first_results = search_function(expanded)
    top_contents = [content for _, content, _ in first_results[:3]]
    feedback_query = pseudo_relevance_feedback(expanded, top_contents)

    final_results = search_function(feedback_query)
    top_docs_content = [content for _, content, _ in final_results[:5]]
    top_docs_names = [name for name, _, _ in final_results[:5]]
    top_score = final_results[0][2] if final_results else 0.0

    if not top_docs_content:
        answer = "No relevant documents found. Please try different keywords."
    else:
        answer = ask_groq(user_question, top_docs_content, groq_api_key)

    return {
        "answer":        answer,
        "top_documents": top_docs_names,
        "total_matches": len(final_results),
        "confidence":    get_confidence_label(top_score),
        "expanded_query": expanded
    }
