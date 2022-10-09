FROM svizor/zoomcamp-model:3.9.12-slim

COPY Pipfile Pipfile.lock service_ml2.py .

RUN pip install pipenv

RUN pipenv install --system

EXPOSE 8001

ENTRYPOINT gunicorn service_ml2:app --bind 0.0.0.0:8001

