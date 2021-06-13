FROM public.ecr.aws/lambda/python:3.8

COPY app.py requirements.txt __init__.py ./
COPY model /opt/ml/model

RUN python3.8 -m pip install -r requirements.txt -t .

CMD ["app.app.lambda_handler"]
