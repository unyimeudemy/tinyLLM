# # Use a lightweight Python base image
# FROM python:3.10-slim

# # Set working directory
# WORKDIR /app

# # Copy requirements file
# COPY requirements.txt .

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy backend files
# COPY main.py .
# COPY llm_1/ llm_1/
# COPY tiny_transformer_weights.pth .

# # Expose FastAPI port
# EXPOSE 8000

# # Run FastAPI with Uvicorn
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# ---- Frontend build ----
FROM node:18 AS frontend
WORKDIR /app/frontend
COPY frontend/ .
RUN npm install && npm run build

# ---- Backend build ----
FROM python:3.11
WORKDIR /app
COPY backend/ .
COPY --from=frontend /app/frontend/dist ./frontend/dist
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
