# FROM python:3.7-buster
FROM 420964472730.dkr.ecr.ap-northeast-1.amazonaws.com/mlops-demo-evaluate:base

ADD . /src
WORKDIR /src

RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3", "/src/preprocess.py"]