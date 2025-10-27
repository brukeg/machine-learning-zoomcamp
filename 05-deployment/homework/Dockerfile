FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

# falling back to trusty old requirements.txt
RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .

ENV MODEL_PATH=/code/pipeline_v2.bin

EXPOSE 9696

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
