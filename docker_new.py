# Use a base image with Python and PyTorch installed
FROM pytorch/pytorch:1.7.0-cuda10.2-cudnn8-devel

# Copy the model code to the container
COPY gpt-3 /app/gpt-3

# Install the required Python packages
RUN pip install -r /app/gpt-3/requirements.txt

# Set the working directory
WORKDIR /app/gpt-3

# Run the model when the container is started
CMD ["python", "run_model.py"]
