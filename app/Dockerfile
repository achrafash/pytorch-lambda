# FROM public.ecr.aws/lambda/python:3.8
FROM public.ecr.aws/bitnami/pytorch:latest

COPY app.py model.py requirements.txt ./
COPY model /opt/ml/model
COPY en_vocab.pkl /opt/ml/en_vocab.pkl
COPY fr_vocab.pkl /opt/ml/fr_vocab.pkl

RUN python3.8 -m pip install -r requirements.txt -t .

CMD ["app.lambda_handler"]
