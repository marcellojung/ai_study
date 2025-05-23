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