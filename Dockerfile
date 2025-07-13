# --- Stage 1: The "Builder" Stage ---
# We use this stage to install all dependencies, including heavy build tools
# and download the ML models. None of this will be in the final image.
FROM python:3.11-slim-bullseye AS builder

# Set the working directory
WORKDIR /app

# Install system dependencies that might be needed for compiling Python packages
# (like python-Levenshtein). We clean up apt cache in the same layer to save space.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment. This is a best practice even inside Docker.
ENV VIRTUAL_ENV=/app/.venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy ONLY the requirements file first to leverage Docker's caching
COPY requirements.txt .

# --- NEW STEP ADDED HERE ---
# Upgrade pip, wheel, and setuptools first. This is crucial for preventing
# installation errors with complex packages like torch and spacy.
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Now, install Python packages into the virtual environment.
RUN pip install --no-cache-dir -r requirements.txt

# --- This is the critical step for spaCy ---
# Download the spaCy model so it's baked into the image.
RUN python -m spacy download en_core_web_sm

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

# Your Procfile specified `worker: python bot.py`.
# This CMD line does the same thing. It tells Docker how to start your bot.
CMD ["python", "bot.py"]