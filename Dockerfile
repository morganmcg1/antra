#FROM python:3.8
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ENV PORT 8501
#EXPOSE 8501

COPY . /app

#ENTRYPOINT [“streamlit”, “run”]
#CMD [“app.py”]

CMD streamlit run app.py
