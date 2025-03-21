import pandas as pd

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import re
import pdfplumber
import torch
from tqdm import tqdm  # ✅ tqdm 추가


train = pd.read_csv('./train.csv', encoding = 'utf-8-sig')
test = pd.read_csv('./test.csv', encoding = 'utf-8-sig')

# 1단계: 사고객체와 작업프로세스를 먼저 채우기
# 공사종류 + 공종을 기준으로 사고객체 채우기
grouped_object = train[train['사고객체'].notnull()].groupby(['공사종류', '공종'])['사고객체'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
train = train.merge(grouped_object, on=['공사종류', '공종'], how='left', suffixes=('', '_filled'))
train['사고객체'] = train['사고객체'].fillna(train['사고객체_filled'])
train.drop(columns=['사고객체_filled'], inplace=True)

# 공사종류 + 공종을 기준으로 작업프로세스 채우기
grouped_process = train[train['작업프로세스'].notnull()].groupby(['공사종류', '공종'])['작업프로세스'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
train = train.merge(grouped_process, on=['공사종류', '공종'], how='left', suffixes=('', '_filled'))
train['작업프로세스'] = train['작업프로세스'].fillna(train['작업프로세스_filled'])
train.drop(columns=['작업프로세스_filled'], inplace=True)

# 2단계: 공종 채우기
# 공사종류 + 사고객체 + 작업프로세스를 기준으로 공종을 채우기
grouped_trade = train[train['공종'].notnull()].groupby(['공사종류', '사고객체', '작업프로세스'])['공종'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
train = train.merge(grouped_trade, on=['공사종류', '사고객체', '작업프로세스'], how='left', suffixes=('', '_filled'))
train['공종'] = train['공종'].fillna(train['공종_filled'])
train.drop(columns=['공종_filled'], inplace=True)

# 3단계: 남은 결측값 처리
# 공사종류만을 기준으로 사고객체 및 작업프로세스 채우기
grouped_object_wo_trade = train[train['사고객체'].notnull()].groupby(['공사종류'])['사고객체'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
train = train.merge(grouped_object_wo_trade, on=['공사종류'], how='left', suffixes=('', '_alt'))
train['사고객체'] = train['사고객체'].fillna(train['사고객체_alt'])
train.drop(columns=['사고객체_alt'], inplace=True)

grouped_process_wo_trade = train[train['작업프로세스'].notnull()].groupby(['공사종류'])['작업프로세스'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
train = train.merge(grouped_process_wo_trade, on=['공사종류'], how='left', suffixes=('', '_alt'))
train['작업프로세스'] = train['작업프로세스'].fillna(train['작업프로세스_alt'])
train.drop(columns=['작업프로세스_alt'], inplace=True)

# 공사종류만을 기준으로 공종 채우기
grouped_trade_wo_object = train[train['공종'].notnull()].groupby(['공사종류'])['공종'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
train = train.merge(grouped_trade_wo_object, on=['공사종류'], how='left', suffixes=('', '_alt'))
train['공종'] = train['공종'].fillna(train['공종_alt'])
train.drop(columns=['공종_alt'], inplace=True)

# 4단계: 남아있는 특정 결측값을 직접 처리
# ID: TRAIN_09122 (공사종류: 건축, 작업프로세스: 설치작업)
train.loc[train['ID'] == 'TRAIN_09122', '공종'] = '건축 > 철근콘크리트공사'
train.loc[train['ID'] == 'TRAIN_09122', '사고객체'] = '건설자재 > 철근'

# ID: TRAIN_21617 (공사종류: 조경, 작업프로세스: 정리작업)
train.loc[train['ID'] == 'TRAIN_21617', '사고객체'] = '기타 > 기타'

# 3단계: 사고원인 자동 생성 (재발방지대책 기반 키워드 매칭)
def infer_accident_cause(row):
    prevention_text = str(row['재발방지대책 및 향후조치계획'])
    
    cause_mapping = {
        '안전조치|안전장비|안전시설': '안전조치 미흡',
        '부주의|주의 부족|주의 미흡': '작업 중 부주의',
        '추락|낙하': '추락방지 미흡',
        '낙석|낙하물': '낙하물 위험',
        '전도': '전도 위험',
        '미끄러짐|넘어짐': '미끄러짐',
        '화재': '화재 위험',
        '중량물|크레인': '중량물 작업 위험',
        '전기|감전': '감전 위험'
    }
    
    for pattern, cause in cause_mapping.items():
        if any(keyword in prevention_text for keyword in pattern.split('|')):
            return cause
    
    return '미상'  # 해당되지 않는 경우 미상으로 처리

# 사고원인 결측값을 자동으로 채우기
train['사고원인'] = train.apply(lambda row: infer_accident_cause(row) if pd.isnull(row['사고원인']) else row['사고원인'], axis=1)


# 1단계: 사고객체와 작업프로세스를 먼저 채우기
# 공사종류 + 공종을 기준으로 사고객체 채우기
grouped_object = test[test['사고객체'].notnull()].groupby(['공사종류', '공종'])['사고객체'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
test = test.merge(grouped_object, on=['공사종류', '공종'], how='left', suffixes=('', '_filled'))
test['사고객체'] = test['사고객체'].fillna(test['사고객체_filled'])
test.drop(columns=['사고객체_filled'], inplace=True)

# 공사종류 + 공종을 기준으로 작업프로세스 채우기
grouped_process = test[test['작업프로세스'].notnull()].groupby(['공사종류', '공종'])['작업프로세스'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
test = test.merge(grouped_process, on=['공사종류', '공종'], how='left', suffixes=('', '_filled'))
test['작업프로세스'] = test['작업프로세스'].fillna(test['작업프로세스_filled'])
test.drop(columns=['작업프로세스_filled'], inplace=True)

# 2단계: 공종 채우기
# 공사종류 + 사고객체 + 작업프로세스를 기준으로 공종을 채우기
grouped_trade = test[test['공종'].notnull()].groupby(['공사종류', '사고객체', '작업프로세스'])['공종'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
test = test.merge(grouped_trade, on=['공사종류', '사고객체', '작업프로세스'], how='left', suffixes=('', '_filled'))
test['공종'] = test['공종'].fillna(test['공종_filled'])
test.drop(columns=['공종_filled'], inplace=True)

# 3단계: 남은 결측값 처리
# 공사종류만을 기준으로 사고객체 및 작업프로세스 채우기
grouped_object_wo_trade = test[test['사고객체'].notnull()].groupby(['공사종류'])['사고객체'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
test = test.merge(grouped_object_wo_trade, on=['공사종류'], how='left', suffixes=('', '_alt'))
test['사고객체'] = test['사고객체'].fillna(test['사고객체_alt'])
test.drop(columns=['사고객체_alt'], inplace=True)

grouped_process_wo_trade = test[test['작업프로세스'].notnull()].groupby(['공사종류'])['작업프로세스'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
test = test.merge(grouped_process_wo_trade, on=['공사종류'], how='left', suffixes=('', '_alt'))
test['작업프로세스'] = test['작업프로세스'].fillna(test['작업프로세스_alt'])
test.drop(columns=['작업프로세스_alt'], inplace=True)

# 공사종류만을 기준으로 공종 채우기
grouped_trade_wo_object = test[test['공종'].notnull()].groupby(['공사종류'])['공종'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
test = test.merge(grouped_trade_wo_object, on=['공사종류'], how='left', suffixes=('', '_alt'))
test['공종'] = test['공종'].fillna(test['공종_alt'])
test.drop(columns=['공종_alt'], inplace=True)

# 데이터 전처리
train['공사종류(대분류)'] = train['공사종류'].str.split(' / ').str[0]
train['공사종류(중분류)'] = train['공사종류'].str.split(' / ').str[1]
train['공종(대분류)'] = train['공종'].str.split(' > ').str[0]
train['공종(중분류)'] = train['공종'].str.split(' > ').str[1]
train['사고객체(대분류)'] = train['사고객체'].str.split(' > ').str[0]
train['사고객체(중분류)'] = train['사고객체'].str.split(' > ').str[1]

test['공사종류(대분류)'] = test['공사종류'].str.split(' / ').str[0]
test['공사종류(중분류)'] = test['공사종류'].str.split(' / ').str[1]
test['공종(대분류)'] = test['공종'].str.split(' > ').str[0]
test['공종(중분류)'] = test['공종'].str.split(' > ').str[1]
test['사고객체(대분류)'] = test['사고객체'].str.split(' > ').str[0]
test['사고객체(중분류)'] = test['사고객체'].str.split(' > ').str[1]

# 훈련 데이터 통합 생성
combined_training_data = train.apply(
    lambda row: {
        "question": (
            f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
            f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
            f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
        ),
        "answer": row["재발방지대책 및 향후조치계획"]
    },
    axis=1
)

# DataFrame으로 변환
combined_training_data = pd.DataFrame(list(combined_training_data))

# 테스트 데이터 통합 생성
combined_test_data = test.apply(
    lambda row: {
        "question": (
            f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
            f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
            f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
        )
    },
    axis=1
)

# DataFrame으로 변환
combined_test_data = pd.DataFrame(list(combined_test_data))


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "beomi/llama-2-koen-13b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

# Train 데이터 준비
train_questions_prevention = combined_training_data['question'].tolist()
train_answers_prevention = combined_training_data['answer'].tolist()

train_documents = [
    f"Q: {q1}\nA: {a1}" 
    for q1, a1 in zip(train_questions_prevention, train_answers_prevention)
]

# 임베딩 생성
embedding_model_name = "jhgan/ko-sbert-nli"  # 임베딩 모델 선택
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

# 벡터 스토어에 문서 추가
vector_store = FAISS.from_texts(train_documents, embedding)

# Retriever 정의
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})


text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,  # sampling 활성화
    temperature=0.1,
    return_full_text=False,
    max_new_tokens=64,
)

prompt_template = """
### 지침: 당신은 건설 안전 전문가입니다.
질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.
- 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.
- 다음과 같은 조치를 취할 것을 제안합니다: 와 같은 내용을 포함하지 마세요.

{context}

### 질문:
{question}

[/INST]

"""

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# 커스텀 프롬프트 생성
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)


# RAG 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,  
    chain_type="stuff",  # 단순 컨텍스트 결합 방식 사용
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}  # 커스텀 프롬프트 적용
)

# 테스트 실행 및 결과 저장
test_results = []

print("테스트 실행 시작... 총 테스트 샘플 수:", len(combined_test_data))

# ✅ tqdm 적용
for idx, row in tqdm(combined_test_data.iterrows(), total=len(combined_test_data), desc="테스트 진행", ncols=100):
    # RAG 체인 호출 및 결과 생성
    prevention_result = qa_chain.invoke(row['question'])

    # 결과 저장
    result_text = prevention_result['result']
    test_results.append(result_text)

print("\n테스트 실행 완료! 총 결과 수:", len(test_results))


embedding_model_name = "jhgan/ko-sbert-sts"
embedding = SentenceTransformer(embedding_model_name)

# 문장 리스트를 입력하여 임베딩 생성
pred_embeddings = embedding.encode(test_results)
print(pred_embeddings.shape)  # (샘플 개수, 768)

submission = pd.read_csv('./sample_submission.csv', encoding = 'utf-8-sig')

# 최종 결과 저장
submission.iloc[:,1] = test_results
submission.iloc[:,2:] = pred_embeddings
submission.head()

# 최종 결과를 CSV로 저장
submission.to_csv('./baseline_submission.csv', index=False, encoding='utf-8-sig')