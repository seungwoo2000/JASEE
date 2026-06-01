import os
import sys
import argparse
from pathlib import Path
import requests
import ssl
from requests.adapters import HTTPAdapter
import urllib3
import warnings
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Suppress console SSL connection warnings and set safe console output encoding
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# Global paths
BASE_DIR = Path(r"C:\python\JASEE")
ENV_PATH = BASE_DIR / ".env"
DB_DIR = BASE_DIR / "vector_db"

# ==========================================
# 1. load_env()
# ==========================================
def load_env():
    load_dotenv(str(ENV_PATH))
    openai_key = os.getenv("OPENAI_API_KEY")
    kdca_key = os.getenv("KDCA_API_KEY")
    model_name = os.getenv("JASEE_MODEL", "gpt-4o")
    return openai_key, kdca_key, model_name

OPENAI_API_KEY, KDCA_API_KEY, JASEE_MODEL = load_env()

# Initialize API clients
if not OPENAI_API_KEY:
    print("[Error] OPENAI_API_KEY not found in .env!")
    sys.exit(1)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
db_client = chromadb.PersistentClient(path=str(DB_DIR))

# Load local embedding model globally to avoid reload latency
print("Loading embedding model (jhgan/ko-sroberta-multitask)...")
embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
print("Embedding model loaded successfully.")

# ==========================================
# 2. get_embedding(text: str) -> list
# ==========================================
def get_embedding(text: str) -> list:
    return embed_model.encode([text])[0].tolist()

# ==========================================
# 3. detect_disease(query: str) -> tuple[str, str]
# ==========================================
def detect_disease(query: str) -> tuple[str, str]:
    query_lower = query.lower()
    
    # Exact keyword to disease mappings
    mappings = [
        {"id": "3796", "name": "요통", "keywords": ["허리", "요통", "허리 통증"]},
        {"id": "3348", "name": "디스크", "keywords": ["디스크", "추간판", "탈출"]},
        {"id": "3628", "name": "척추측만증", "keywords": ["척추측만", "측만증", "척추 휘어짐"]},
        {"id": "3629", "name": "척추후만증", "keywords": ["척추후만", "후만증", "등 굽음"]},
        {"id": "5972", "name": "거북목", "keywords": ["거북목", "일자목", "목 통증", "경추"]},
        {"id": "6292", "name": "수근굴증후군", "keywords": ["손목", "수근굴", "수근관", "저림"]}
    ]
    
    for m in mappings:
        for kw in m["keywords"]:
            if kw in query_lower:
                return m["id"], m["name"]
                
    # Default fallback
    return "5972", "거북목"

# ==========================================
# 4. classify_function(query: str) -> list[str]
# ==========================================
def classify_function(query: str, default_func_id: int = 3) -> list[str]:
    query_lower = query.lower()
    
    # 키워드 우선순위 규칙 추가
    func1_priority_kws = ["각도", "tia", "cva", "측정", "판정"]
    func4_priority_kws = ["통증", "아파", "뻐근"]
    
    has_func1_kw = any(kw in query_lower for kw in func1_priority_kws)
    has_func4_kw = any(kw in query_lower for kw in func4_priority_kws)
    
    if has_func1_kw:
        # "각도", "TIA", "CVA", "측정", "판정" 키워드가 있으면 무조건 기능1 우선 (동시 감지 시에도 기능1 우선)
        return ["jasee_func1"]
    elif has_func4_kw:
        # "통증", "아파", "뻐근"만 있으면 기능4
        return ["jasee_func4"]
        
    # Keyword mapping
    keywords = {
        "jasee_func1": ["각도", "tia", "cva", "판정", "good", "bad", "자세 분석", "측정", "피드백"],
        "jasee_func2": ["rula", "점수", "위험도", "몇 점", "위험 수준", "평가"],
        "jasee_func3": ["작업환경", "의자", "모니터", "키보드", "조명", "높이", "법", "고시", "책상", "환경"],
        "jasee_func4": ["통증", "아파", "뻐근", "저려", "경추", "요추", "손목", "목", "허리", "질환"],
        "jasee_func5": ["운동", "스트레칭", "교정", "강화", "패턴", "ucs", "lcs", "flatback", "swayback"]
    }
    
    scores = {}
    for coll_name, kw_list in keywords.items():
        score = 0
        for kw in kw_list:
            if kw in query_lower:
                score += 1
        if score > 0:
            scores[coll_name] = score
            
    if not scores:
        return [f"jasee_func{default_func_id}"]
        
    sorted_colls = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_colls) >= 2:
        return [sorted_colls[0][0], sorted_colls[1][0]]
    else:
        return [sorted_colls[0][0]]

# ==========================================
# 5. search_chunks(query: str, collections: list, top_k=5) -> list
# ==========================================
def search_chunks(query: str, collections: list, top_k=5) -> list:
    query_vector = get_embedding(query)
    all_results = []
    
    for coll_name in collections:
        try:
            coll = db_client.get_collection(name=coll_name)
            k = 3 if len(collections) > 1 else top_k
            res = coll.query(
                query_embeddings=[query_vector],
                n_results=k
            )
            
            if res and res.get("documents") and len(res["documents"][0]) > 0:
                docs = res["documents"][0]
                metas = res["metadatas"][0]
                distances = res["distances"][0] if res.get("distances") else [0.0] * len(docs)
                
                for d, m, dist in zip(docs, metas, distances):
                    all_results.append({
                        "content": d,
                        "source_file": m.get("source_file", "N/A"),
                        "chunk_id": m.get("chunk_id", "N/A"),
                        "distance": dist,
                        "collection": coll_name
                    })
        except Exception:
            pass
            
    all_results = sorted(all_results, key=lambda x: x["distance"])
    return all_results

# ==========================================
# 6. build_prompt(function_id: str, query: str, chunks: list) -> str
# ==========================================
def build_prompt(function_id: str, query: str, chunks: list) -> str:
    system_prompts = {
        "1": ("당신은 자세 분석 전문가입니다.\n"
              "측정값과 판정 이유를 중심으로 설명하세요.\n"
              "교정 방법은 1줄 이내만 언급하세요."),
        "2": ("당신은 RULA 평가 전문가입니다.\n"
              "점수가 어느 위험 수준인지,\n"
              "어느 부위 점수가 총점을 올렸는지 설명하세요.\n"
              "교정 방법은 언급하지 마세요."),
        "3": ("당신은 작업환경 전문가입니다.\n"
              "즉시 실행 가능한 구체적 수치를 제시하고\n"
              "VDT 고시 또는 KOSHA GUIDE 조항 근거를 반드시 포함하세요."),
        "4": ("당신은 근골격계 전문 상담사입니다.\n"
              "경추·요추·손목 증상별로\n"
              "스트레칭과 완화법을 중심으로 답변하세요."),
        "5": ("당신은 재활 운동 전문가입니다.\n"
              "UCS/LCS/FlatBack/SwayBack 패턴을 먼저 확인하고\n"
              "tight/weak 근육에 맞는 운동을 구체적으로 제공하세요.")
    }
    
    sys_prompt = system_prompts.get(str(function_id), "당신은 인체공학 전문가 챗봇입니다.")
    
    context_lines = []
    for idx, c in enumerate(chunks):
        context_lines.append(f"[문서 {idx+1}] (출처: {c['source_file']} | ID: {c['chunk_id']})\n{c['content']}")
    context_str = "\n\n".join(context_lines)
    
    prompt = (
        f"System Instruction:\n{sys_prompt}\n\n"
        f"Search Contexts:\n{context_str}\n\n"
        f"User Query:\n{query}"
    )
    return prompt

# ==========================================
# 7. call_llm(prompt: str, history: list) -> str
# ==========================================
def call_llm(prompt: str, history: list) -> str:
    system_instruction = "당신은 인체공학 전문가 챗봇입니다."
    user_content = prompt
    
    if "System Instruction:\n" in prompt:
        parts = prompt.split("System Instruction:\n", 1)[1].split("\n\nSearch Contexts:\n", 1)
        if len(parts) == 2:
            system_instruction = parts[0].strip()
            user_content = "[검색된 컨텍스트]\n" + parts[1]

    messages = [
        {"role": "system", "content": system_instruction}
    ]
    
    for turn in history[-5:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
        
    messages.append({"role": "user", "content": user_content})
    
    try:
        response = openai_client.chat.completions.create(
            model=JASEE_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"오류: OpenAI GPT-4o API 호출 중 예외가 발생했습니다 ({e})."

# ==========================================
# 8. call_kdca_api(query: str) -> list[dict] | None
# ==========================================
def call_kdca_api(query: str) -> list[dict] | None:
    try:
        # 1. 키워드로 질환 ID 감지
        disease_id, disease_name = detect_disease(query)
        print(f"[KDCA API] 질환 감지: {disease_name}")
        print("[KDCA API] 실시간 데이터 조회 중...")
        
        # 2. 해당 질환 URL + TOKEN 로드
        url = os.environ.get(f"KDCA_URL_{disease_id}_{disease_name}")
        token = os.environ.get(f"KDCA_TOKEN_{disease_id}_{disease_name}")
        
        if not url:
            print("[KDCA API] 실패 → JSON 폴백")
            return None
            
        # 3. API 호출
        params = {"serviceKey": token}
        
        # Setup Legacy SSL Adapter to handle legacy handshake requirements
        class LegacySSLAdapter(HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                ctx = ssl.create_default_context()
                ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                kwargs['ssl_context'] = ctx
                return super(LegacySSLAdapter, self).init_poolmanager(*args, **kwargs)
                
        session = requests.Session()
        session.mount('https://', LegacySSLAdapter())
        session.mount('http://', LegacySSLAdapter())
        
        response = session.get(url, params=params, verify=False, timeout=5)
        
        if response.status_code in [200, 201]:
            # 4. 응답 파싱 → 청크 리스트 반환
            root = ET.fromstring(response.text)
            chunks = []
            for cl in root.findall('.//cntntsCl'):
                cl_nm_el = cl.find('CNTNTS_CL_NM')
                cl_cn_el = cl.find('CNTNTS_CL_CN')
                if cl_nm_el is not None and cl_cn_el is not None:
                    cl_nm = cl_nm_el.text.strip() if cl_nm_el.text else ""
                    cl_cn = cl_cn_el.text.strip() if cl_cn_el.text else ""
                    if cl_cn and not cl_cn.startswith("http"):
                        content = f"### {cl_nm}\n{cl_cn}"
                        chunks.append({
                            "content": content,
                            "source_file": "질병관리청 건강정보포털 (실시간)",
                            "chunk_id": f"ID: {disease_id} {disease_name}",
                            "distance": 0.0
                        })
            if chunks:
                print("[KDCA API] 조회 성공")
                return chunks
                
        print("[KDCA API] 실패 → JSON 폴백")
        return None
    except Exception:
        print("[KDCA API] 실패 → JSON 폴백")
        return None

# ==========================================
# 9. RAG CORE PROCESSOR & FUNCTION 4 FALLBACK
# ==========================================
def process_rag_query(query: str, selected_func_id: int, history: list) -> str:
    collections = classify_function(query, default_func_id=selected_func_id)
    chunks = search_chunks(query, collections, top_k=5)
    
    is_func4_mode = (selected_func_id == 4 or "jasee_func4" in collections)
    is_poor_match = (not chunks or all(c["distance"] > 0.7 for c in chunks))
    
    if is_func4_mode and is_poor_match:
        api_chunks = call_kdca_api(query)
        if api_chunks:
            chunks = api_chunks
        else:
            disease_id, disease_name = detect_disease(query)
            return ("관련 정보를 찾지 못했습니다. 전문의 상담을 권장합니다.\n\n"
                    f"[출처] 질병관리청 건강정보포털 (실시간) | ID: {disease_id} {disease_name}")

    if not chunks:
        return "관련 정보를 데이터베이스에서 찾지 못했습니다."

    prompt = build_prompt(str(selected_func_id), query, chunks)
    answer = call_llm(prompt, history)
    
    # Format footer strictly: [출처] source_file | chunk_id
    footers = []
    for c in chunks:
        footers.append(f"{c['source_file']} | {c['chunk_id']}")
    unique_footers = []
    for f in footers:
        if f not in unique_footers:
            unique_footers.append(f)
            
    source_footer = "\n\n[출처] " + ", ".join(unique_footers)
    return answer + source_footer

# ==========================================
# 10. receive_from_team()
# ==========================================
def receive_from_team(measurement_dict: dict) -> dict:
    """
    팀원 코드(OpenCV/YOLO-pose) 연동 인터페이스
    팀원 코드 완성 후 내부 구현 예정

    입력 예시:
    {
        "TIA": 25.3,
        "CVA": 18.7,
        "elbow": 95.0,
        "wrist": 10.0,
        "knee": 90.0,
        "monitor": 12.0
    }

    출력 예시:
    {
        "feedback": "피드백 텍스트",
        "bad_indicators": ["TIA"],
        "good_indicators": ["CVA"],
        "reference": "VDT 고시 제6조",
        "exercise_recommend": True
    }
    """
    pass  # TODO: 팀원 코드 받은 후 구현

# ==========================================
# 11. run_chatbot() - Terminal Menu Loop
# ==========================================
def run_chatbot():
    print("\n========================================")
    print("     자세히봐(Jasee) RAG 챗봇 v1.0")
    print("========================================")
    
    while True:
        print("기능을 선택하세요:")
        print("  [1] 자세 분석 결과 설명")
        print("  [2] RULA 점수 해석")
        print("  [3] 작업환경 적합성")
        print("  [4] 부위별 통증 완화")
        print("  [5] 운동 추천")
        print("  [0] 종료")
        print("========================================")
        
        try:
            choice_str = input("\033[96m선택> \033[0m").strip()
            if choice_str == "0":
                print("\n자세히봐(Jasee) 챗봇 서비스를 종료합니다. 올바른 자세로 건강한 하루 되세요!")
                break
            if choice_str not in ["1", "2", "3", "4", "5"]:
                print("\033[91m올바른 번호를 선택하세요 (1~5 또는 0).\033[0m\n")
                continue
                
            selected_func_id = int(choice_str)
            func_titles = {
                1: "자세 분석 결과 설명 모드",
                2: "RULA 점수 해석 모드",
                3: "작업환경 적합성 진단 모드",
                4: "부위별 통증 완화 상담 모드",
                5: "개인화 맞춤 운동 추천 모드"
            }
            
            print(f"\n\033[94m--- [{selected_func_id}] {func_titles[selected_func_id]} 진입 ---")
            print("자유롭게 질문해 주세요! (메뉴로 돌아가려면 'q' 또는 'menu' 입력)\033[0m\n")
            
            history = []
            
            while True:
                query = input("\033[92m나: \033[0m").strip()
                if not query:
                    continue
                if query.lower() in ["q", "menu", "quit", "exit"]:
                    print("\n메인 메뉴로 돌아갑니다.\n")
                    history = []
                    break
                    
                print("\033[90m자세 전문가 답변을 작성하는 중...\033[0m")
                answer = process_rag_query(query, selected_func_id, history)
                print(f"\n\033[93mJasee 챗봇:\033[0m\n{answer}\n")
                print("-" * 50)
                
                clean_ans = answer.split("\n\n[출처]")[0]
                history.append({
                    "user": query,
                    "assistant": clean_ans
                })
                
        except KeyboardInterrupt:
            print("\n자세히봐(Jasee) 챗봇 세션을 강제 종료합니다.")
            break
        except Exception as e:
            print(f"\n\033[91m예상치 못한 오류 발생: {e}\033[0m\n")

# ==========================================
# 12. CLI DRIVER
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jasee RAG Chatbot CLI")
    parser.add_argument("--query", type=str, help="Executes a single test query and exits.")
    parser.add_argument("--func_id", type=int, default=3, help="Default function ID context.")
    args = parser.parse_args()
    
    if args.query:
        detected_colls = classify_function(args.query, default_func_id=args.func_id)
        best_func_id = args.func_id
        if detected_colls:
            try:
                best_func_id = int(detected_colls[0].replace("jasee_func", ""))
            except:
                pass
                
        answer = process_rag_query(args.query, best_func_id, history=[])
        print(answer)
    else:
        run_chatbot()
