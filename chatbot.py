import os, re
from collections import Counter
from groq import Groq
from gensim.models import Word2Vec
 
 
# ════════════════════════════════════════════════════════════
# SECTION 1 — THESAURUS  (Lecture 7)
# ════════════════════════════════════════════════════════════
 
THESAURUS = {
    "e coli":           ["escherichia coli", "coliform", "gram negative rod"],
    "ecoli":            ["escherichia coli", "coliform", "e coli"],
    "salmonella":       ["salmonellosis", "typhoid", "gram negative enterobacteria"],
    "cholera":          ["vibrio cholerae", "waterborne disease", "gram negative"],
    "legionella":       ["legionnaires disease", "waterborne pathogen", "gram negative"],
    "pseudomonas":      ["gram negative rod", "aerobic bacteria", "opportunistic pathogen"],
    "staph":            ["staphylococcus", "gram positive cocci", "s aureus", "mrsa"],
    "staphylococcus":   ["staph", "gram positive cocci", "s aureus", "mrsa"],
    "strep":            ["streptococcus", "gram positive cocci"],
    "mrsa":             ["methicillin resistant staphylococcus", "antibiotic resistant staph"],
    "gram negative":    ["gram-negative", "gnb", "enterobacteriaceae"],
    "gram positive":    ["gram-positive", "gpb", "firmicutes"],
    "treatment":        ["therapy", "antibiotic", "antimicrobial", "medication"],
    "antibiotic":       ["antimicrobial", "antibacterial", "drug"],
    "infection":        ["contamination", "disease", "pathogen"],
    "water":            ["aquatic", "waterborne", "drinking water", "contamination"],
    "resistant":        ["antibiotic resistance", "drug resistant", "multidrug", "alternative treatment"],
    "resistance":       ["resistant bacteria", "multidrug resistant", "alternative antibiotic"],
}
 
def expand_query(query):
    query_lower    = query.lower()
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
 
 
# ════════════════════════════════════════════════════════════
# SECTION 2 — WORD2VEC  (Lecture 8)
# ════════════════════════════════════════════════════════════
 
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
        sentences   = tokenized,
        vector_size = 100,
        window      = 5,
        min_count   = 1,
        sg          = 1,
        epochs      = 30,
        workers     = 2
    )
    print(f"[Word2Vec] Trained on {len(tokenized)} docs. Vocabulary: {len(model.wv)} words.")
    return model
 
USELESS = {
    'are', 'and', 'the', 'was', 'has', 'been', 'have', 'that', 'this',
    'with', 'from', 'they', 'were', 'also', 'found', 'more', 'than',
    'not', 'for', 'but', 'its', 'can', 'may', 'such', 'some', 'most',
    'inhabitants', 'habitats', 'copepods', 'regions', 'hours', 'cells',
    'your', 'symptoms', 'sources', 'rates', 'levels', 'types', 'cases'
}
 
def get_word2vec_expansions(query, w2v_model, top_n=3, threshold=0.80):
    words       = re.findall(r'\b[a-z]{3,}\b', query.lower())
    extra_words = []
    for word in words:
        if word in w2v_model.wv:
            similar = w2v_model.wv.most_similar(word, topn=top_n)
            for sim_word, score in similar:
                if (score >= threshold
                        and sim_word not in query.lower()
                        and sim_word not in USELESS):
                    extra_words.append(sim_word)
    return list(dict.fromkeys(extra_words))
 
 
# ════════════════════════════════════════════════════════════
# SECTION 3 — PSEUDO RELEVANCE FEEDBACK  (Lecture 7)
# ════════════════════════════════════════════════════════════
 
STOPWORDS = {
    'that', 'this', 'with', 'from', 'they', 'were', 'have', 'been',
    'their', 'also', 'which', 'when', 'some', 'more', 'will', 'used',
    'such', 'most', 'than', 'into', 'other', 'these', 'those', 'both',
    'about', 'after', 'before', 'between', 'your', 'symptoms', 'hours',
    'vibrio', 'cholerae', 'cholera', 'salmonella', 'pseudomonas',
    'legionella', 'staphylococcus', 'streptococcus', 'enterococcus',
    'klebsiella'
}
 
def pseudo_relevance_feedback(original_query, top_docs_content, top_k_words=3):
    if not top_docs_content:
        return original_query
    combined    = " ".join(top_docs_content[:2]).lower()
    words       = re.findall(r'\b[a-z]{4,}\b', combined)
    words       = [w for w in words if w not in STOPWORDS]
    freq        = Counter(words)
    query_words = set(original_query.lower().split())
    new_words   = []
    for word, _ in freq.most_common(20):
        if word not in query_words and len(new_words) < top_k_words:
            new_words.append(word)
    if new_words:
        print(f"  [Feedback] Added words: {new_words}")
    return original_query + " " + " ".join(new_words)
 
 
# ════════════════════════════════════════════════════════════
# SECTION 3B — RERANKER
# ════════════════════════════════════════════════════════════
 
BACTERIA_TOPIC_MAP = {
    'coli':           ['coli', 'ecoli', 'escherichia'],
    'escherichia':    ['coli', 'ecoli', 'escherichia'],
    'coliform':       ['coli', 'ecoli', 'escherichia'],
    'salmonella':     ['salmonella'],
    'typhoid':        ['salmonella', 'typhoid'],
    'pseudomonas':    ['pseudomonas', 'aeruginosa'],
    'aeruginosa':     ['pseudomonas', 'aeruginosa'],
    'legionella':     ['legionella', 'legionnaires'],
    'legionnaires':   ['legionella', 'legionnaires'],
    'staph':          ['staph', 'staphylococcus'],
    'staphylococcus': ['staph', 'staphylococcus'],
    'mrsa':           ['mrsa', 'methicillin', 'staphylococcus'],
    'strep':          ['strep', 'streptococcus'],
    'streptococcus':  ['strep', 'streptococcus'],
    'klebsiella':     ['klebsiella'],
    'enterobacter':   ['enterobacter'],
    'cholera':        ['cholera', 'vibrio'],
    'vibrio':         ['vibrio', 'cholera'],
    'listeria':       ['listeria'],
    'campylobacter':  ['campylobacter'],
    'shigella':       ['shigella', 'dysentery'],
}
 
def rerank_results(user_question, results):
    query_lower = user_question.lower()
    topic_words = []
    for query_kw, doc_keywords in BACTERIA_TOPIC_MAP.items():
        if query_kw in query_lower:
            topic_words.extend(doc_keywords)
    if not topic_words:
        return results
    print(f"  [Reranker] Topic: {list(set(topic_words))}")
    scored = []
    for name, content, original_score in results:
        name_lower     = name.lower()
        content_lower  = content.lower()[:500]
        filename_bonus = 0.3 if any(kw in name_lower    for kw in topic_words) else 0.0
        content_bonus  = 0.2 if any(kw in content_lower for kw in topic_words) else 0.0
        scored.append((name, content, original_score + filename_bonus + content_bonus))
    scored.sort(key=lambda x: x[2], reverse=True)
    print(f"  [Reranker] New top 3: {[n for n,_,_ in scored[:3]]}")
    return scored
 
 
# ════════════════════════════════════════════════════════════
# SECTION 3C — RELEVANT CHUNK EXTRACTOR
# ════════════════════════════════════════════════════════════
 
def get_relevant_chunk(doc_content, query, chunk_size=700):
    query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
    paragraphs  = [p.strip() for p in doc_content.split('\n') if len(p.strip()) > 60]
    if not paragraphs:
        return doc_content[:chunk_size]
    best_para  = doc_content[:chunk_size]
    best_score = -1
    for para in paragraphs:
        para_words = set(re.findall(r'\b[a-z]{3,}\b', para.lower()))
        score      = len(query_words & para_words)
        if score > best_score:
            best_score = score
            best_para  = para
    return best_para[:chunk_size]
 
 
# ════════════════════════════════════════════════════════════
# SECTION 4 — GROQ API  (RAG)
# ════════════════════════════════════════════════════════════
 
conversation_history = []
 
def ask_groq(user_question, relevant_docs, api_key):
    global conversation_history
    context = "\n\n---\n\n".join([
        get_relevant_chunk(doc, user_question)
        for doc in relevant_docs[:3]
    ])
    system_message = {
        "role": "system",
        "content": f"""You are a medical and water safety assistant specialized in waterborne bacterial infections.
Answer questions about bacteria, water contamination, and treatments.
 
RULES:
- Answer naturally and directly.
- Do NOT say 'the documents say' or 'in the provided documents'.
- NEVER guess or add information from outside the knowledge base below.
- If the knowledge base does not cover the question, say exactly:
  'I don't have enough information on this. Please consult a microbiologist or refer to WHO guidelines.'
 
KNOWLEDGE BASE:
{context}"""
    }
    conversation_history.append({"role": "user", "content": user_question})
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model       = "llama-3.1-8b-instant",
        messages    = [system_message] + conversation_history,
        temperature = 0.1,
        max_tokens  = 500
    )
    answer = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": answer})
    return answer
 
 
def reset_conversation():
    global conversation_history
    conversation_history = []
    print("Conversation reset.")
 
 
# ════════════════════════════════════════════════════════════
# SECTION 5 — MAIN FUNCTION
# ════════════════════════════════════════════════════════════
 
def get_answer(user_question, search_function, docs_folder, groq_api_key, w2v_model=None):
    search_query  = user_question
    user_messages = [m for m in conversation_history if m["role"] == "user"]
    if user_messages:
        last_user_msg = user_messages[-1]["content"]
        search_query  = last_user_msg + " " + user_question
        print(f"[Context] Added previous question to search")
 
    print(f"\n{'='*55}")
    print(f"Question: {user_question}")
    print(f"{'='*55}")
 
    # Step 1: Thesaurus expansion
    expanded = expand_query(search_query)
    print(f"[Step 1] Thesaurus expansion done")
 
    # Step 2: Word2Vec expansion
    if w2v_model is not None:
        w2v_words = get_word2vec_expansions(expanded, w2v_model)
        if w2v_words:
            expanded = expanded + " " + " ".join(w2v_words)
            print(f"[Step 2] Word2Vec added: {w2v_words}")
        else:
            print(f"[Step 2] Word2Vec: no new terms")
    else:
        print(f"[Step 2] Word2Vec: skipped")
 
    # Step 3: First search + rerank
    raw_first     = search_function(expanded)
    first_results = rerank_results(user_question, raw_first)
    top_contents  = [content for _, content, _ in first_results[:3]]
    top_names     = [name    for name,  _, _ in first_results[:3]]
    print(f"[Step 3] First search top docs: {top_names}")
 
    # Step 4: Pseudo relevance feedback
    feedback_query = pseudo_relevance_feedback(expanded, top_contents)
 
    # Step 5: Second search + rerank
    raw_final        = search_function(feedback_query)
    final_results    = rerank_results(user_question, raw_final)
    top_docs_content = [content for _, content, _ in final_results[:5]]
    top_docs_names   = [name    for name,  _, _ in final_results[:5]]
    print(f"[Step 5] Final search top docs: {top_docs_names}")
 
    # Step 6: Groq generates answer
    print(f"[Step 6] Asking Groq LLM...")
    if not top_docs_content:
        answer = "No relevant documents found. Try different keywords."
    else:
        answer = ask_groq(user_question, top_docs_content, groq_api_key)
 
    print(f"[Done] Answer generated!")
 
    return {
        "answer":         answer,
        "expanded_query": expanded,
        "top_documents":  top_docs_names,
        "total_matches":  len(final_results)
    }
 