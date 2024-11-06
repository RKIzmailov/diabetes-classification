FROM python:3.11.5

WORKDIR /

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app /bins /data

COPY ./data/* ./data/
COPY ./bins/model.bin ./bins/
COPY ./app/predict.py ./app/
COPY ./app/test.py /app/

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--host=0.0.0.0", "--port=9696", "app.predict:app"]