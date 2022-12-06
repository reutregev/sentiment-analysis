FROM python:3.8

WORKDIR /sentiment-analysis

COPY ./requirements.txt /sentiment-analysis/requirements.txt
RUN pip install -r /sentiment-analysis/requirements.txt

COPY . /sentiment-analysis/

ENV PYTHONPATH=/sentiment-analysis/

EXPOSE 9980

CMD ["python", "/sentiment-analysis/app/app.py", "--model_dir", "/sentiment-analysis/model/trained_model.pt"]
