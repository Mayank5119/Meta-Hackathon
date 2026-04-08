# Use official Python 3.11 slim image as the base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# (We use --no-cache-dir to keep the image size small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the FastAPI server runs on
EXPOSE 7060

# Command to run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7060"]