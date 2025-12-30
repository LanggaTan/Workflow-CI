# Use a Miniconda base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Create the Conda environment from conda.yaml
RUN conda env create -f conda.yaml

# Make sure the environment is activated in the container
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

# Install MLflow in the environment (optional if not in conda.yaml)
RUN pip install mlflow

# Default command to run MLflow project
CMD ["mlflow", "run", "."]
