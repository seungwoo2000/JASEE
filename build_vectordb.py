import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

# Set standard output encoding to UTF-8 for safe console logging on Windows
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# 1. PATH CONFIGURATIONS
# ==========================================
BASE_DIR = Path(r"C:\python\JASEE")
DATA_DIR = BASE_DIR / "processed_data"
DB_DIR = BASE_DIR / "vector_db"
DB_DIR.mkdir(parents=True, exist_ok=True)

# Define collection-to-file mapping
COLLECTIONS_MAP = {
    "jasee_func1": "function1_chunks.json",
    "jasee_func2": "function2_chunks.json",
    "jasee_func3": "function3_chunks.json",
    "jasee_func4": "function4_chunks.json",
    "jasee_func5": "function5_chunks.json"
}

# ==========================================
# 2. CUSTOM EMBEDDING FUNCTION wrapping jhgan/ko-sroberta-multitask
# ==========================================
class KoSRobertaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="jhgan/ko-sroberta-multitask"):
        print(f"Loading embedding model '{model_name}' (this might take a moment on first download)...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully.")

    def __call__(self, input: Documents) -> Embeddings:
        # Encode documents and convert the resulting numpy array to nested lists of floats
        embeddings = self.model.encode(input, show_progress_bar=False)
        return embeddings.tolist()

# ==========================================
# 3. VECTOR DB BUILD PIPELINE
# ==========================================
def build_vector_db():
    print("Initializing ChromaDB Persistent Client...")
    # Initialize the local persistent client
    client = chromadb.PersistentClient(path=str(DB_DIR))
    
    # Initialize the custom embedding function
    embedding_fn = KoSRobertaEmbeddingFunction()

    counts = {}
    grand_total = 0

    print("\nStarting indexing process for all collections...")

    for coll_name, file_name in COLLECTIONS_MAP.items():
        file_path = DATA_DIR / file_name
        if not file_path.exists():
            print(f"[WARNING] Source file {file_name} not found! Skipping collection '{coll_name}'.")
            counts[coll_name] = 0
            continue

        print(f"\nProcessing Collection '{coll_name}' from {file_name}...")
        
        # Load processed chunks
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        if not chunks:
            print(f"  -> Collection '{coll_name}' has 0 chunks. Skipping.")
            counts[coll_name] = 0
            continue

        # Get or create the isolated collection
        collection = client.get_or_create_collection(
            name=coll_name,
            embedding_function=embedding_fn
        )

        # Batch indexing setup (100 chunks per batch)
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            
            ids = []
            documents = []
            metadatas = []

            for chunk in batch:
                ids.append(str(chunk["chunk_id"]))
                documents.append(str(chunk["content"]))
                
                # Format metadata safely for ChromaDB constraints (no lists allowed)
                meta = {
                    "chunk_id": str(chunk.get("chunk_id", "")),
                    "source_file": str(chunk.get("source_file", "")),
                    "doc_type": str(chunk.get("doc_type", "")),
                    "function": ",".join(chunk.get("function", [])),
                    "body_part": ",".join(chunk.get("body_part", []))
                }
                
                # Optional metadata parameters (omit if None to keep DB clean)
                if chunk.get("rula_score_range") is not None:
                    meta["rula_score_range"] = str(chunk["rula_score_range"])
                if chunk.get("tia_angle") is not None:
                    meta["tia_angle"] = str(chunk["tia_angle"])
                if chunk.get("posture_indicator") is not None:
                    meta["posture_indicator"] = str(chunk["posture_indicator"])
                    
                # Include function 4 & 5 specific fields if present
                if chunk.get("vdt_category") is not None:
                    meta["vdt_category"] = str(chunk["vdt_category"])
                if chunk.get("vdt_source") is not None:
                    meta["vdt_source"] = str(chunk["vdt_source"])
                if chunk.get("pattern_id") is not None:
                    meta["pattern_id"] = str(chunk["pattern_id"])
                if chunk.get("muscle_type") is not None:
                    meta["muscle_type"] = str(chunk["muscle_type"])

                metadatas.append(meta)

            # Upsert into ChromaDB collection
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

        coll_count = len(chunks)
        print(f"  -> Successfully indexed {coll_count} chunks in collection '{coll_name}'.")
        counts[coll_name] = coll_count
        grand_total += coll_count

    # Print final collection status report
    print("\n==========================================")
    print("Vector Database Build Completed Successfully!")
    print("==========================================")
    for coll_name, count in counts.items():
        print(f"{coll_name}: {count}개")
    print(f"전체 합계: {grand_total}개")
    print("==========================================")

if __name__ == "__main__":
    build_vector_db()
