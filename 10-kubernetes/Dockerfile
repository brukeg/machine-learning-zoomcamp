FROM python:3.12-slim-bookworm

# Install Pipenv
RUN pip install --no-cache-dir pipenv

WORKDIR /code

# Keep virtualenv inside project (like your uv setup) and expose it on PATH
ENV PIPENV_VENV_IN_PROJECT=1
ENV PATH="/code/.venv/bin:$PATH"

# Copy dependency manifests from the *build context root* (repo root)
COPY Pipfile Pipfile.lock ./
RUN pipenv install --deploy --ignore-pipfile

# Copy app code from the subfolder
COPY 10-kubernetes/app.py 10-kubernetes/clothing-model.onnx ./

EXPOSE 8080
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
