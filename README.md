# ai_study
ai study : ml, dl, rag, peft.


## 🔧 설치 & 실행

> **Python 3.9+** 환경을 권장합니다.  
> GPU가 있다면 TensorFlow·PyTorch 연산이 빨라집니다.

```bash
# 1) 저장소 클론
git clone <your-repo-url>
cd <repo>

# 2) 가상환경 생성 (선택)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) 필수 패키지 설치
pip install -r requirements.txt
```





## 파일 및 핵심내용, 주요라이브러리 


| 파일 | 핵심 내용 | 주요 라이브러리 |
|------|-----------|----------------|
| `ref_Konlpy_의미기반형태소분석기.ipynb` | KoNLPy 형태소 분석기(Okt·Komoran·Hannanum·Kkma) 사용법과 의미 기반 토큰화 실습 | `konlpy`, `pandas` |
| `ref_언어모델간임베딩유사도비교.ipynb` | Word2Vec·FastText·Transformer·OpenAI Embedding 간 벡터 유사도(cosine) 비교 | `gensim`, `sentence_transformers`, `openai`, `scikit-learn`, `numpy` |
| `ref_언어모델별_단어예측예시.ipynb` | N-gram, RNN, FastText, Word2Vec 등을 이용한 단어 예측(다음 단어 생성) 예시 | `konlpy`, `fasttext`, `gensim`, `tensorflow`, `matplotlib` |
| `ref_Langchain을_사용한것과_안한것의_차이.ipynb` | LangChain을 통한 언어모델 통합 관리와 직접 API 호출 방식의 비교 | `langchain`, `openai`, `anthropic`, `requests` |
| `1_OPEN_API_기초.ipynb` | OpenAI API 기본 사용법과 텍스트 생성 실습 | `openai`, `tiktoken` |
| `2_GPT기반_온라인_댓글_분류_자동.ipynb` | GPT를 활용한 온라인 댓글 자동 분류 시스템 구현 | `openai`, `pandas` |
| `3_GPT_BASIC_stock.ipynb` | GPT를 활용한 주식 데이터 분석 및 예측 | `openai`, `pandas`, `yfinance` |
| `4_FunctionCalling_OPENAI_Assistant구현.ipynb` | OpenAI Assistant API의 Function Calling 기능 구현 | `openai` |
| `5_OPENAI_Assistant구현.ipynb` | OpenAI Assistant API를 활용한 챗봇 구현 | `openai` |
| `6_LangChain_basic.ipynb` | LangChain 기본 개념과 활용법 | `langchain`, `openai` |
| `7_1_LangChain을_이용한_댓글_분류_자동화.ipynb` | LangChain을 활용한 댓글 자동 분류 시스템 | `langchain`, `openai`, `pandas` |
| `7_2_LangChain을_이용한_답글_생성_자동화.ipynb` | LangChain을 활용한 자동 답글 생성 시스템 | `langchain`, `openai` |
| `8_벡터_데이터베이스_기반_RAG어플리케이션.ipynb` | 벡터 DB를 활용한 RAG(Retrieval-Augmented Generation) 구현 | `langchain`, `chromadb`, `openai` |
| `9_PDF내용_기반_질의응답_애플리케이션.ipynb` | PDF 문서 기반 질의응답 시스템 구현 | `langchain`, `pypdf`, `openai` |
| `10_LangChain을_이용한_SQL_Database분석.ipynb` | LangChain을 활용한 SQL 데이터베이스 분석 | `langchain`, `sqlalchemy`, `openai` |
| `13_fine_tunning.ipynb` | 언어모델 파인튜닝 기초와 실습 | `transformers`, `datasets`, `torch` |
| `14_llama2_파인튜닝.ipynb` | LLaMA2 모델 파인튜닝 실습 | `transformers`, `peft`, `torch` |


##  🔎 RAG 기반 파인튜닝 과정 (Retrieval-Augmented Generation)

> **RAG**는 사전학습된 LLM에 문서 검색 기능(Retriever)을 결합하여, 최신 정보나 도메인 지식 기반으로 답변하는 방식입니다.

---

### 📋 전체 단계 요약

| 단계 | 주요 내용 | 핵심 포인트 |
|------|-----------|-------------|
| **1. 문서 수집 및 정제** | PDF, TXT, HTML 등 다양한 원문 | 불필요한 정보 제거, 문단 분리, 텍스트 정규화 |
| **2. 문서 분할 (Chunking)** | 문서를 작은 단위로 나누기 | 고정 길이 또는 문맥 기반 분할 (ex. 512 tokens) |
| **3. 임베딩 생성** | 문서 → 벡터화 | `sentence-transformers`, `OpenAI`, `Hugging Face` 임베딩 모델 사용 |
| **4. 벡터 DB 구축** | FAISS, Weaviate, Qdrant 등 | 빠른 유사도 검색 지원, 메타정보와 함께 저장 |
| **5. Retriever 구성** | 쿼리 입력 → 관련 문서 검색 | similarity search / dense retrieval 방식 선택 |
| **6. Generation 모델 연결** | 검색 결과 + 쿼리 → LLM 입력 | `RAG`, `LLMChain`, `custom prompt` 설계 필요 |
| **7. 파인튜닝(Optional)** | RAG pipeline 전체 또는 LLM만 | 도메인 특화 답변 개선, `LoRA` 또는 `SFT` 활용 |
| **8. 평가 및 배포** | Answer 정확도 평가, Gradio, API | 실사용 시 latency 고려, 버전 관리 필요 |

---

### 🧠 핵심 구성 요소

- 🔹 **Retriever (Dense/Hybrid)**: 유사 문서 검색 담당
- 🔹 **Generator (LLM)**: 최종 응답 생성
- 🔹 **Vector DB**: 문서 벡터를 저장하고 검색

---

### 🔧 주요 사용 라이브러리

- `langchain`, `llama-index`, `transformers`
- `faiss`, `chromadb`, `qdrant`, `weaviate`
- `sentence-transformers`, `openai-embeddings`
- `gradio`, `fastapi`, `streamlit` (UI/API)

---

### ✅ 실전 팁

- 문서 분할은 **질문에 답할 수 있을 정도의 단위**로 유지
- 임베딩 모델은 질문과 문서 유형에 맞게 선택
- LLM 파인튜닝 없이도 RAG 정확도를 높일 수 있음 (Prompt + Chunking 전략)

---

## 🔧 파인튜닝 과정 (Fine-Tuning Process)

> 사전학습된 LLM을 **내가 원하는 데이터에 맞게 다시 학습시키는 과정**입니다.

---

### 📋 전체 단계 요약

| 단계 | 주요 내용 | 핵심 포인트 |
|------|-----------|-------------|
| **1. 사전학습 모델 선택** | LLaMA, Mistral 등 | 오픈소스 여부, 모델 크기(B 단위), 토크나이저 호환 |
| **2. 데이터셋 준비** | 질문-응답 형식 (Alpaca 등) | 포맷 변환 (JSON, ChatML), 전처리 필요 |
| **3. 토크나이저 설정** | 텍스트 → 토큰 변환 | 모델과 동일한 토크나이저 사용 (예: LLaMA → LlamaTokenizer) |
| **4. 파인튜닝 방식 선택** | Full, LoRA, QLoRA | 자원 상황에 따라 선택 (LoRA 추천) |
| **5. 하이퍼파라미터 설정** | `learning_rate`, `batch_size` 등 | 실험적으로 조정 필요, 모델/데이터 크기 기반 |
| **6. 학습 수행** | `Trainer`, `SFTTrainer` 등 | GPU/TPU 활용, 체크포인트 저장 및 WandB/로깅 활용 |
| **7. 모델 저장 및 배포** | `save_pretrained()`, Hugging Face Hub 업로드 | 추론 API 개발 가능, 로컬 또는 Hub 배포 |

---

### 📦 참고 사항

- **LoRA/QLoRA** 방식은 GPU 자원이 부족한 경우 매우 효율적입니다.
- Hugging Face `transformers`, `peft`, `datasets`, `accelerate` 패키지를 활용하여 구축 가능합니다.
- 실험 반복이 필요하므로 **WandB** 또는 로깅 도구 사용을 권장합니다.


### llama_7B_koalpaca_sft_lora_merge 디렉토리 구조 설명

| 파일명 | 설명 |
|--------|------|
| `config.json` | 모델 구조 및 하이퍼파라미터 설정 파일입니다. (예: hidden size, num_layers 등)<br>모델 로드 시 필요한 메타정보를 담고 있습니다. |
| `generation_config.json` | 텍스트 생성에 사용되는 설정값 (예: `max_length`, `temperature`, `top_k`, `top_p`)이 담겨 있습니다. |
| `model-00001-of-00002.safetensors`<br>`model-00002-of-00002.safetensors` | 모델의 실제 가중치가 저장된 파일입니다. `.safetensors` 포맷은 안전하고 빠른 로딩을 위한 Hugging Face 표준 포맷입니다. |
| `model.safetensors.index.json` | 여러 개로 나뉜 `.safetensors` 파일들을 연결해주는 인덱스 파일입니다. |
| `special_tokens_map.json` | `[PAD]`, `[CLS]`, `[SEP]` 등 특수 토큰의 ID 매핑 정보를 담고 있습니다. |
| `tokenizer.json` | fast tokenizer가 사용하는 주요 토크나이저 정의 파일입니다. (vocab, merges 포함) |
| `tokenizer_config.json` | 토크나이저 동작 방식에 대한 설정값 (lowercase 여부, special tokens 등)이 담긴 파일입니다. |

---

### LLAMA 3.1 + RAG vs LLAMA 3.1 + Fine-tuning 비교 

### 1. 실험 결과 비교
- **LLAMA 3.1 + RAG**  
- **LLAMA 3.1 + Fine-tuning**  
→ 두 방식 간 성능 차이 없음  
→ **RAG**가 더 효율적

---

### 2. RAG 성능 향상 방안
1. **Chain-of-Thought (CoT) 기반 프롬프트 엔지니어링**  
   - 복잡한 추론 과정을 단계별로 유도하여 성능 개선  
2. **테이블 처리 성능 보완**  
   - 기본적으로 텍스트는 왼쪽→오른쪽으로,  
     표(table)는 위→아래로 읽어야 하므로 처리 방식 차이로 성능 저하 발생  

---

### 3. 테이블 읽기 개선: `unstructured` 라이브러리
- **적용 배경**  
  - RAG에서 테이블 인식 및 파싱 정확도가 낮음  
- **도입 효과**  
  - `unstructured` 라이브러리 추가  
  - 표 구조를 위→아래 흐름으로 정확히 파악  
  - **테이블 처리 성능 대폭 향상**

---