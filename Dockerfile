# --- Stage 1: The Builder ---
# Use a slim Python image as a base for building our dependencies.
# Pinning the version ensures consistent builds.
FROM python:3.10-slim-bullseye AS builder

# Set the working directory inside the container
WORKDIR /app

# Set environment variables for pip and Python
ENV PIP_NO_CACHE_DIR=off \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a virtual environment. This is a best practice for managing dependencies.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy ONLY the requirements file first to leverage Docker's layer caching.
# This way, dependencies are only re-installed if requirements.txt changes.
COPY requirements.txt .

# Install dependencies into the virtual environment.
# THIS IS THE MOST IMPORTANT PART:
# --index-url https://download.pytorch.org/whl/cpu tells pip to get the CPU-only version of torch.
# This will reduce the torch installation size from gigabytes to a few hundred megabytes.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

# Download the spaCy model into the virtual environment's site-packages.
RUN python -m spacy download en_core_web_sm


# --- Stage 2: The Final Image ---
# Use the same slim base image for the final, lean container.
FROM python:3.10-slim-bullseye

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage.
# This gives us all the installed packages without any of the build tools.
COPY --from=builder /opt/venv /opt/venv

# Copy the application code into the final image
COPY . .

# Set the PATH to use the virtual environment's Python and packages
ENV PATH="/opt/venv/bin:$PATH"

# Run the bot as a non-root user for better security
RUN useradd --create-home appuser
USER appuser

# The command to run when the container starts.
# Railway will override this with the command from your railway.json, but it's good practice to have it.
CMD ["python", "bot.py"]