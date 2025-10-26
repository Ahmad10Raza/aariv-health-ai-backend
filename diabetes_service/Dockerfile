# Use lightweight Python 3.12 image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app

# Expose port for the API
EXPOSE 5000

# Run Flask app with Gunicorn (3 workers)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "3", "app:app"]
