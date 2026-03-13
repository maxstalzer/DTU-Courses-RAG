import json
import os
import re
import unicodedata
import numpy as np
from contextlib import asynccontextmanager
from typing import List, Literal, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from dotenv import load_dotenv
import bm25s
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from fastapi.concurrency import run_in_threadpool

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# CONFIGURATION & GLOBAL VARIABLES
# ---------------------------------------------------------
DATA_PATH = "data/dtu_courses.jsonl" 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

llm_client: AsyncOpenAI = None
embed_model: SentenceTransformer = None

course_data = []      

sparse_courses_title = bm25s.BM25()
sparse_courses_fields = bm25s.BM25()
sparse_courses_obj = bm25s.BM25()

dense_embs_courses = None

# --- ACCURACY FIX: Text Normalization for BM25 ---
def clean_text(text: str) -> str:
    """Lowercases text and strips accents (e.g., Bjørn -> bjorn)"""
    if not text: return ""
    # Normalize unicode and encode to ASCII to drop weird characters
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text.lower()

# ---------------------------------------------------------
# STARTUP / LIFESPAN
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client, embed_model, dense_embs_courses, course_data

    llm_client = AsyncOpenAI(
        api_key=GROQ_API_KEY, 
        base_url="https://api.groq.com/openai/v1"
    )

    print("Loading BGE-Small Sentence-Transformer model...")
    embed_model = SentenceTransformer("intfloat/multilingual-e5-small")

    print("Loading course data...")
    courses_list = []
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    courses_list.append(json.loads(line))
    
    for info in courses_list:
        cid = info.get("course_code", "")
        title = info.get("title", "")
        
        fields_dict = info.get("fields", {})
        fields_clean = ", ".join([f"{k}: {v}" for k, v in fields_dict.items() if v])
        
        objectives = info.get("learning_objectives", [])
        content = info.get("content", "")
        objs_str = " ".join(objectives)
        
        course_text = f"{title}\n{fields_clean}\n{objs_str}\n{content}"
        
        course_data.append({
            "id": cid, 
            "title": title, 
            "fields": fields_clean, 
            "objs": objs_str,
            "content": content, # Store content so we can retrieve it later
            "text": course_text 
        })

    print("Initializing BM25 Indices with stripped accents...")
    if course_data:
        # Pass cleaned text into the indexers
        sparse_courses_title.index(bm25s.tokenize([clean_text(c["title"]) for c in course_data]))
        sparse_courses_fields.index(bm25s.tokenize([clean_text(c["fields"]) for c in course_data]))
        sparse_courses_obj.index(bm25s.tokenize([clean_text(c["objs"]) for c in course_data]))
        
        course_texts = [c["text"] for c in course_data]
        
        cache_file = "data/dense_courses_intfloat.npy"
        if os.path.exists(cache_file):
            print("Loading local Course embeddings from disk cache...")
            dense_embs_courses = np.load(cache_file)
        else:
            print("Encoding courses locally with multilingual intfloat...")
            os.makedirs("data", exist_ok=True)
            dense_embs_courses = embed_model.encode(course_texts, normalize_embeddings=True, show_progress_bar=True)
            np.save(cache_file, dense_embs_courses)

    print("System ready!")
    yield
    await llm_client.close()

app = FastAPI(title="DTU Course RAG API", lifespan=lifespan)

# ---------------------------------------------------------
# ASYNC RETRIEVAL LOGIC
# ---------------------------------------------------------
async def get_search_scores(query_text: str, mode: str, alpha: float) -> np.ndarray:
    # --- FIX 1: Pad 4-digit numbers with a zero (e.g. 2451 -> 02451) ---
    query_text = re.sub(r'\b(\d{4})\b', r'0\g<1>', query_text)
    
    if dense_embs_courses is None:
        return np.array([])
    
    sparse_scores = np.zeros(dense_embs_courses.shape[0])
    if mode in ["sparse", "hybrid"]:
        # Clean the user query before searching BM25
        clean_query = clean_text(query_text)
        query_words = re.findall(r'\w+', clean_query)
        
        def normalize(scores_array):
            max_val = scores_array.max()
            return scores_array / max_val if max_val > 0 else scores_array

        if query_words:
            score_t = normalize(sparse_courses_title.get_scores(query_words))
            score_f = normalize(sparse_courses_fields.get_scores(query_words))
            score_o = normalize(sparse_courses_obj.get_scores(query_words))
            sparse_scores = (score_t * 0.4) + (score_f * 0.4) + (score_o * 0.2)

    dense_scores = np.zeros(dense_embs_courses.shape[0])
    if mode in ["dense", "hybrid"]:
        # e5-small requires "query: " before the search text
        e5_query = "query: " + query_text 
        query_dense = await run_in_threadpool(
            embed_model.encode, [e5_query], normalize_embeddings=True
        )
        # BUG FIX: Restored the actual matrix multiplication calculation!
        dense_scores = (query_dense @ dense_embs_courses.T).flatten()

    if mode == "dense":
        return dense_scores
    elif mode == "sparse":
        return sparse_scores
    else: 
        return (alpha * dense_scores) + ((1 - alpha) * sparse_scores)

# ---------------------------------------------------------
# ASYNC ENDPOINTS
# ---------------------------------------------------------
@app.get("/v1/search")
async def search_courses(
    query: str,
    top_k: int = Query(15, ge=1, le=50), # Default increased to 15
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
    alpha: float = Query(0.90, ge=0.0, le=1.0) # Boosted to 0.90 to survive typos
):
    scores = await get_search_scores(query, mode, alpha)
    
    top_indices_unordered = np.argpartition(-scores, top_k)[:top_k]
    top_indices = top_indices_unordered[np.argsort(-scores[top_indices_unordered])]
    
    results = [
        {"course_code": course_data[i]["id"], "title": course_data[i]["title"], "score": round(float(scores[i]), 3)} 
        for i in top_indices
    ]
    return {"query": query, "mode": mode, "results": results}


@app.get("/v1/ask")
async def ask_question(
    query: str,
    top_k: int = Query(10, ge=1, le=15), 
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
    alpha: float = Query(0.90, ge=0.0, le=1.0) # Boosted to 0.90
):
    scores = await get_search_scores(query, mode, alpha)
    
    top_indices_unordered = np.argpartition(-scores, top_k)[:top_k]
    top_indices = top_indices_unordered[np.argsort(-scores[top_indices_unordered])]
    
    retrieved_courses = []
    context_blocks = []
    
    for i in top_indices:
        course = course_data[i]
        retrieved_courses.append({"course_code": course["id"], "title": course["title"]})
        
        objs_safe = course['objs'][:600] + "..." if len(course['objs']) > 600 else course['objs']
        
        # --- FIX 2: Head & Tail Chunking ---
        # Grab the first 400 chars AND the last 400 chars to catch the schedule at the bottom!
        full_content = course['content']
        if len(full_content) > 800:
            content_safe = full_content[:400] + "\n...[TRUNCATED]...\n" + full_content[-400:]
        else:
            content_safe = full_content

        block = f"[{course['id']} - {course['title']}]\n"
        block += f"Details: {course['fields']}\n"
        block += f"Objectives: {objs_safe}\n"
        block += f"Content: {content_safe}"
        context_blocks.append(block)

    full_context = "\n\n".join(context_blocks)
    
    prompt = f"""You are a helpful DTU student advisor. Answer the question based ONLY on the context below. 
If the context does not contain the answer, explicitly say you do not know. 
CRITICAL INSTRUCTIONS: 
1. Always answer in a complete, standalone sentence.
2. If asked for a list (like "which courses"), you MUST list ALL matching courses found in the context. Do not stop at just one.

Context:
{full_context}

Question: {query}
Answer:"""

    response = await llm_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0 
    )
    
    return {
        "query": query,
        "answer": response.choices[0].message.content.strip(),
        "retrieved_courses": retrieved_courses
    }