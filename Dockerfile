# --- Stage 1: The "Builder" Stage ---
# We use this stage to install all dependencies, including heavy build tools
# and download the ML models. None of this will be in the final image.
FROM python:3.11-slim-bullseye AS builder

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment. This is a best practice even inside Docker.
ENV VIRTUAL_ENV=/app/.venv
RUN python3 -m venv $VIRTUAL_ENV
# We still set the PATH for the final image's CMD command
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy ONLY the requirements file first to leverage Docker's caching
COPY requirements.txt .

# --- EXPLICIT PATH FIX ---
# Upgrade pip using its full path inside the virtual environment
RUN /app/.venv/bin/pip install --no-cache-dir --upgrade pip wheel setuptools

# --- EXPLICIT PATH FIX ---
# Install Python packages using the venv's pip
RUN /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# --- EXPLICIT PATH FIX ---
# Download the spaCy model using the venv's python
RUN /app/.venv/bin/python -m spacy download en_core_web_sm

# Copy the rest of your application's source code
COPY . .


# --- Stage 2: The "Final" Lean Stage ---
# This is the actual image that will be deployed. It's clean and minimal.
FROM python:3.11-slim-bullseye AS final

WORKDIR /app

# Set the same virtual environment path
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the virtual environment (with all installed packages and the spaCy model)
# from the builder stage. This is the magic of multi-stage builds.
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy your application code from the builder stage.
COPY --from=builder /app /app

# The CMD command will correctly use the PATH we set, so this doesn't need a full path.
CMD ["python", "bot.py"]