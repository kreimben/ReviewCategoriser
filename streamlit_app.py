import os

import boto3
import streamlit as st
import torch
from dotenv import load_dotenv
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from torch_related.module import RoBERTaFineTuner

load_dotenv()

token = os.getenv('HUGGINGFACE_TOKEN')
boto_access_key = os.getenv('BOTO_ACCESS_KEY')
boto_secret_key = os.getenv('BOTO_SECRET_KEY')


@st.cache_resource
def load_roberta_model(token):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', token=token)
    roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4, token=token)
    return tokenizer, roberta_model


@st.cache_resource
def load_model(device):
    s3 = boto3.client(
        's3',
        aws_access_key_id=boto_access_key,
        aws_secret_access_key=boto_secret_key,
        region_name='ap-northeast-2'
    )
    s3.download_file('kreimben-general-bucket', 'trained_models/2403040608model.ckpt', 'model.ckpt')
    model = RoBERTaFineTuner.load_from_checkpoint('model.ckpt', model=roberta_model, device=device.type)
    return model


tokenizer, roberta_model = load_roberta_model(token)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

model = load_model(device)

# model = RoBERTaFineTuner.load_from_checkpoint('2403040608model.ckpt', model=roberta_model, device=device.type)
label_map = {0: "informational", 1: "happy", 2: 'profanity', 3: 'anger'}


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return label_map[predicted_class_id]


st.title("RoBERTa 기반 후기 텍스트 분류기")
st.write('''
후기 데이터를 입력해 주세요. 결과는 4가지 카테고리로 나옵니다.\n
`informational`, `happy`, `profanity`, `anger` 총 4가지 카테고리가 있습니다.\n
''')
st.write('''
예시 데이터)\n
이 음식점은 서비스가 매우 친절하고 음식 퀄리티도 좋아요. => `informational`\n
영화를 보고 나왔을 때 기분이 너무 좋았어요. 강추합니다! => `happy`\n
이 쇼핑몰은 주문 후 배송이 너무 늦어서 짜증나네요. 다신 이용 안 할 거에요. => `profanity`\n
관광지에 가서 사람이 너무 많아서 정말 짜증났어요. 한 번 더 갈 일 없을 것 같아요. => `anger`\n
''')
user_input = st.text_area("Input Text", "후기 데이터를 입력해 주세요.")
if st.button("predict"):
    prediction = predict(user_input)
    st.write(f"예측 결과: `{prediction}` 가 분류 결과입니다.")
