FROM python:3

WORKDIR /app

COPY *.py *.txt *.pth .

RUN ["pip", "install", "-r", "requirements-prod.txt"]

ENTRYPOINT ["python", "app.py"]