FROM python:3.9

RUN pip install pipenv uvicorn fastapi joblib scikit-learn

WORKDIR /app

COPY ./Pipfile.lock /app/Pipfile.lock

COPY ./main.py /app/
COPY ./svm.ml /app/
COPY ./linear.ml /app/

CMD ["pipenv",  "run",  "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]