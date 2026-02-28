"""
Semantic RAG - Premium Web Interface
=====================================
A stunning, production-grade web interface for the semantic search module.

Design: Dark luxury theme with golden accents
Aesthetic: Refined, professional, memorable

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import psycopg2
import time
from sentence_transformers import SentenceTransformer

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Semantic RAG | Rose Blanche",
    page_icon="🌹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== PREMIUM CSS ====================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Source+Sans+Pro:wght@300;400;600&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* ===== GLOBAL STYLES ===== */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
        background-attachment: fixed;
    }
    
    /* Noise texture overlay */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        opacity: 0.03;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
        pointer-events: none;
        z-index: 0;
    }
    
    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: #f5f5f5 !important;
    }
    
    p, span, div, label {
        font-family: 'Source Sans Pro', sans-serif !important;
    }
    
    code {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* ===== MAIN HEADER ===== */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 1rem;
        position: relative;
    }
    
    .main-header::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 120px;
        height: 2px;
        background: linear-gradient(90deg, transparent, #d4af37, transparent);
    }
    
    .logo-text {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #d4af37 0%, #f5d77a 50%, #d4af37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 40px rgba(212, 175, 55, 0.3);
    }
    
    .tagline {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1.1rem;
        color: #888;
        letter-spacing: 4px;
        text-transform: uppercase;
        font-weight: 300;
    }
    
    .org-badge {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.5rem 1.5rem;
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 30px;
        font-size: 0.85rem;
        color: #d4af37;
        letter-spacing: 2px;
        background: rgba(212, 175, 55, 0.05);
    }
    
    /* ===== SEARCH BOX ===== */
    .search-container {
        max-width: 900px;
        margin: 2rem auto;
        padding: 2.5rem;
        background: linear-gradient(145deg, rgba(30, 30, 45, 0.8), rgba(20, 20, 35, 0.9));
        border-radius: 20px;
        border: 1px solid rgba(212, 175, 55, 0.2);
        box-shadow: 
            0 25px 50px -12px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .search-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(212, 175, 55, 0.5), transparent);
    }
    
    .search-label {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        color: #f5f5f5;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .search-label::before {
        content: "◆";
        color: #d4af37;
        font-size: 0.8rem;
    }
    
    /* Style Streamlit input */
    .stTextInput > div > div > input {
        font-family: 'Source Sans Pro', sans-serif !important;
        font-size: 1.15rem !important;
        padding: 1rem 1.5rem !important;
        background: rgba(10, 10, 20, 0.6) !important;
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
        border-radius: 12px !important;
        color: #f5f5f5 !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #d4af37 !important;
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.2) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #666 !important;
        font-style: italic;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        background: linear-gradient(135deg, #d4af37 0%, #aa8a2e 100%) !important;
        color: #0a0a0f !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        font-size: 0.9rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(212, 175, 55, 0.3) !important;
        background: linear-gradient(135deg, #e5c04b 0%, #d4af37 100%) !important;
    }
    
    /* Quick action buttons */
    .quick-btn {
        display: inline-block;
        padding: 0.6rem 1.2rem;
        margin: 0.3rem;
        background: rgba(212, 175, 55, 0.1);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 8px;
        color: #d4af37;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .quick-btn:hover {
        background: rgba(212, 175, 55, 0.2);
        transform: translateY(-1px);
    }
    
    /* ===== RESULT CARDS ===== */
    .results-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: #f5f5f5;
        margin: 2rem 0 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(212, 175, 55, 0.2);
    }
    
    .result-card {
        background: linear-gradient(145deg, rgba(25, 25, 40, 0.9), rgba(15, 15, 28, 0.95));
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .result-card::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #d4af37, #aa8a2e);
    }
    
    .result-card:hover {
        transform: translateX(8px);
        border-color: rgba(212, 175, 55, 0.2);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    .result-rank {
        position: absolute;
        top: 1rem;
        right: 1rem;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #d4af37, #aa8a2e);
        border-radius: 50%;
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: #0a0a0f;
    }
    
    .result-text {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1.05rem;
        line-height: 1.7;
        color: #d0d0d0;
        margin-bottom: 1rem;
        padding-right: 50px;
    }
    
    .result-text::before {
        content: open-quote;
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: #d4af37;
        opacity: 0.5;
        margin-right: 0.3rem;
    }
    
    .result-text::after {
        content: close-quote;
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: #d4af37;
        opacity: 0.5;
        margin-left: 0.3rem;
    }
    
    .result-score {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 1rem;
        background: rgba(212, 175, 55, 0.1);
        border-radius: 20px;
        border: 1px solid rgba(212, 175, 55, 0.3);
    }
    
    .score-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .score-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        color: #d4af37;
    }
    
    .score-bar {
        width: 100px;
        height: 4px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 2px;
        overflow: hidden;
        margin-left: 0.5rem;
    }
    
    .score-fill {
        height: 100%;
        background: linear-gradient(90deg, #d4af37, #f5d77a);
        border-radius: 2px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12121a 0%, #0a0a0f 100%) !important;
        border-right: 1px solid rgba(212, 175, 55, 0.1) !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #d4af37 !important;
    }
    
    /* ===== METRICS ===== */
    .metric-card {
        background: rgba(20, 20, 35, 0.8);
        border: 1px solid rgba(212, 175, 55, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(212, 175, 55, 0.4);
        transform: translateY(-3px);
    }
    
    .metric-value {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 700;
        color: #d4af37;
        display: block;
    }
    
    .metric-label {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .animate-fade-in {
        animation: fadeInUp 0.6s ease forwards;
    }
    
    .animate-delay-1 { animation-delay: 0.1s; }
    .animate-delay-2 { animation-delay: 0.2s; }
    .animate-delay-3 { animation-delay: 0.3s; }
    
    /* ===== STATUS INDICATORS ===== */
    .status-online {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.3rem 0.8rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 20px;
        font-size: 0.8rem;
        color: #10b981;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    /* ===== DIVIDERS ===== */
    .gold-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(212, 175, 55, 0.4), transparent);
        margin: 2rem 0;
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(212, 175, 55, 0.1);
    }
    
    .footer-text {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.9rem;
        color: #555;
        letter-spacing: 1px;
    }
    
    .footer-logo {
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        color: #d4af37;
        margin-bottom: 0.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==================== CONFIGURATION ====================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "database",
    "user": "user",
    "password": "password"
}

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

# ==================== CACHED RESOURCES ====================

@st.cache_resource
def load_model():
    """Load the embedding model (cached)."""
    return SentenceTransformer(MODEL_NAME)

@st.cache_data(ttl=300)
def load_embeddings():
    """Load embeddings from database (cached)."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id, id_document, texte_fragment, vecteur
            FROM embeddings ORDER BY id
        """)
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        if not rows:
            return [], np.array([]), []
        
        fragments, vectors, doc_ids = [], [], []
        
        for row in rows:
            doc_ids.append(row[1])
            fragments.append(row[2])
            vec = row[3]
            if isinstance(vec, str):
                import ast
                vec = ast.literal_eval(vec)
            vectors.append(np.array(vec, dtype=np.float32))
        
        return fragments, np.array(vectors), doc_ids
        
    except Exception as e:
        return [], np.array([]), []

# ==================== SEARCH FUNCTIONS ====================

def cosine_similarity(vec_a, vec_b):
    norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)

def semantic_search(query, model, fragments, vectors, top_k=TOP_K):
    if len(vectors) == 0:
        return []
    
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    scores = np.array([cosine_similarity(query_embedding, vec) for vec in vectors])
    sorted_indices = np.argsort(scores)[::-1]
    
    results, seen = [], set()
    for idx in sorted_indices:
        fragment = fragments[idx]
        if fragment not in seen:
            seen.add(fragment)
            results.append({"texte": fragment, "score": float(scores[idx])})
            if len(results) >= top_k:
                break
    return results

# ==================== UI COMPONENTS ====================

def render_header():
    st.markdown("""
    <div class="main-header">
        <div class="logo-text">Semantic RAG</div>
        <div class="tagline">Intelligence Artificielle pour la Recherche Documentaire</div>
        <div class="org-badge">🌹 Rose Blanche Group</div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar(fragments):
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 3rem;">🌹</span>
            <h2 style="font-family: 'Playfair Display', serif; color: #d4af37; margin-top: 0.5rem;">
                Rose Blanche
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        
        # Status
        st.markdown("""
        <div class="status-online">
            <span class="status-dot"></span>
            Système Actif
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">{len(fragments)}</span>
                <span class="metric-label">Fragments</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">384</span>
                <span class="metric-label">Dimensions</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 1rem;">
            <span class="metric-value" style="font-size: 1rem;">all-MiniLM-L6-v2</span>
            <span class="metric-label">Modèle IA</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <h3 style="color: #d4af37; font-size: 1rem;">📋 Spécifications</h3>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <ul style="color: #888; font-size: 0.9rem; line-height: 2;">
            <li>Similarité Cosinus</li>
            <li>Top K = 3 résultats</li>
            <li>PostgreSQL + pgvector</li>
            <li>Embeddings normalisés</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: rgba(212, 175, 55, 0.05); border-radius: 10px; border: 1px solid rgba(212, 175, 55, 0.2);">
            <div style="font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px;">Challenge</div>
            <div style="font-family: 'Playfair Display', serif; color: #d4af37; font-size: 1.2rem; margin: 0.5rem 0;">RoboKids 84</div>
            <div style="font-size: 1.5rem;">🏆</div>
            <div style="color: #d4af37; font-weight: 600;">1000 DT</div>
        </div>
        """, unsafe_allow_html=True)

def render_result_card(result, index):
    score = result['score']
    score_percent = min(score * 100, 100)
    
    st.markdown(f"""
    <div class="result-card animate-fade-in animate-delay-{index}">
        <div class="result-rank">{index}</div>
        <div class="result-text">{result['texte']}</div>
        <div class="result-score">
            <span class="score-label">Score</span>
            <span class="score-value">{score:.2f}</span>
            <div class="score-bar">
                <div class="score-fill" style="width: {score_percent}%;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    st.markdown("""
    <div class="footer">
        <div class="footer-logo">🌹 Rose Blanche Group</div>
        <div class="footer-text">
            Module de Recherche Sémantique (RAG) — Challenge RoboKids 2024
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN APPLICATION ====================

def main():
    # Render header
    render_header()
    
    # Load resources
    with st.spinner(""):
        model = load_model()
        fragments, vectors, doc_ids = load_embeddings()
    
    # Render sidebar
    render_sidebar(fragments)
    
    # Initialize session state for query
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    # Main content
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="search-label">Posez votre question</div>', unsafe_allow_html=True)
    
    # Quick action buttons FIRST (before text input)
    st.markdown("<span style='color: #666; font-size: 0.9rem;'>Questions suggérées:</span>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💊 Dosage alpha-amylase", use_container_width=True):
            st.session_state.query = "Quel est le dosage recommandé d'alpha-amylase ?"
    
    with col2:
        if st.button("🍞 Rôle de la xylanase", use_container_width=True):
            st.session_state.query = "Comment la xylanase améliore la pâte ?"
    
    with col3:
        if st.button("🧪 Acide ascorbique", use_container_width=True):
            st.session_state.query = "Rôle de l'acide ascorbique dans la panification ?"
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Search input
    query = st.text_input(
        "Votre question",
        value=st.session_state.query,
        placeholder="Ex: Quel est le dosage recommandé d'alpha-amylase pour la farine ?",
        key="search_input",
        label_visibility="collapsed"
    )
    
    # Update session state with typed query
    if query != st.session_state.query:
        st.session_state.query = query
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Results
    if query:
        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="results-header">
            Résultats pour « {query[:50]}{'...' if len(query) > 50 else ''} »
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Recherche sémantique en cours..."):
            time.sleep(0.3)  # Small delay for effect
            results = semantic_search(query, model, fragments, vectors)
        
        if results:
            for i, result in enumerate(results, 1):
                render_result_card(result, i)
            
            # Summary metrics
            avg_score = sum(r['score'] for r in results) / len(results)
            max_score = max(r['score'] for r in results)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <span class="metric-value">{len(results)}</span>
                    <span class="metric-label">Résultats</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <span class="metric-value">{avg_score:.2f}</span>
                    <span class="metric-label">Score Moyen</span>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <span class="metric-value">{max_score:.2f}</span>
                    <span class="metric-label">Meilleur Score</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Aucun résultat trouvé pour cette requête.")
    
    # Footer
    render_footer()

if __name__ == "__main__":
    main()
