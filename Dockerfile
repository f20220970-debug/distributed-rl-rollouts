# docker build -t drl . && docker run --rm -v "$PWD/results:/app/results" -v "$PWD/docs:/app/docs" drl
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy gymnasium matplotlib pytest \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY . .
CMD ["sh", "-c", "pytest -q && python run_experiments.py"]
