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





## deep learning


| 파일 | 핵심 내용 | 주요 라이브러리 |
|------|-----------|----------------|
| `ref_Konlpy_의미기반형태소분석기.ipynb` | KoNLPy 형태소 분석기(Okt·Komoran·Hannanum·Kkma) 사용법과 의미 기반 토큰화 실습 | `konlpy`, `pandas` |
| `ref_언어모델간임베딩유사도비교.ipynb` | Word2Vec·FastText·Transformer·OpenAI Embedding 간 벡터 유사도(cosine) 비교 | `gensim`, `sentence_transformers`, `openai`, `scikit-learn`, `numpy` |
| `ref_언어모델별_단어예측예시.ipynb` | N-gram, RNN, FastText, Word2Vec 등을 이용한 단어 예측(다음 단어 생성) 예시 | `konlpy`, `fasttext`, `gensim`, `tensorflow`, `matplotlib` |
