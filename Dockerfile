# Start from a slim official Python image. We pin 3.12 deliberately:
# it's well-tested and we control it here regardless of the host machine.
FROM python:3.12-slim

# Set the working directory inside the container. Everything below runs here.
WORKDIR /app

# Copy requirements FIRST, then install. This ordering matters for build
# caching: as long as requirements.txt doesn't change, Docker reuses the
# cached install layer on rebuilds instead of reinstalling every time.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code and the trained model bundle.
COPY api.py .
COPY models/ ./models/

# Document which port the service listens on.
EXPOSE 8000

# The command that runs when the container starts. Note host 0.0.0.0, not
# 127.0.0.1: inside a container the service must bind to all interfaces so
# it's reachable from outside the container.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]