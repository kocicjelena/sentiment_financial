FROM python:3.10.8-slim
LABEL description="Sentiment machine learning"
WORKDIR /usr/src/app
COPY requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt
COPY . /usr/src/app/
CMD uvicorn app:app --host 0.0.0.0 --port 5000
#CMD ["python",  "app.py"]
EXPOSE 5000