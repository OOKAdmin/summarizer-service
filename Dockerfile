FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Hugging Face Spaces binds to port 7860 by default
EXPOSE 7860
ENV PORT=7860

# Start the Flask app
CMD ["python", "app.py"]
