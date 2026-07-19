# Use an official lightweight Python image
FROM python:3.10-slim

# Install system dependencies required for Git and building dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose port 7860 (Hugging Face Spaces expects port 7860)
EXPOSE 7860

# Run the Flask app using Gunicorn bound to port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "600", "app:app"]