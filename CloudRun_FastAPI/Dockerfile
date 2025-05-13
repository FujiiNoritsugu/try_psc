# Use Python 3.10 slim as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry==1.7.1

# Copy poetry configuration files
COPY pyproject.toml ./

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY app/ ./app/

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 8080

# Run the application
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
