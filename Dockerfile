# --- Stage 1: The Builder ---
# Use a slim Python image as a base for building our dependencies.
FROM python:3.10-slim-bullseye AS builder

# Set the working directory inside the container
WORKDIR /app

# Set environment variables for pip and Python
ENV PIP_NO_CACHE_DIR=off \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --- NEW: Install system build dependencies ---
# We need these to compile some Python packages (like python-Levenshtein)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy ONLY the requirements file first to leverage Docker's layer caching.
COPY requirements.txt .

# --- UPDATED: Install Python packages ---
# 1. Upgrade pip
# 2. Install torch CPU separately to ensure the correct version is used.
# 3. Install the rest of the packages from requirements.txt.
RUN pip install --upgrade pip && \
    pip install torch --no-cache-dir --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Download the spaCy model
RUN python -m spacy download en_core_web_sm


# --- Stage 2: The Final Image ---
# Use the same slim base image for the final, lean container.
FROM python:3.10-slim-bullseye

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Copy the application code into the final image
COPY . .

# Set the PATH to use the virtual environment's Python and packages
ENV PATH="/opt/venv/bin:$PATH"

# Run the bot as a non-root user for better security
RUN useradd --create-home appuser
USER appuser

# The command to run when the container starts.
CMD ["python", "bot.py"]