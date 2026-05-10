FROM python:3.12.0-slim
# Set the working directory in the container

WORKDIR /app
# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Expose
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

#Specify the command to run Fastapi
CMD ["uvicorn", "apis.main:app", "--host", "0.0.0.0", "--port", "8000"]