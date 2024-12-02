# Use an official Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements_inference.txt /app/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_inference.txt

# Copy the app code into the container
COPY . /app/

# Initialize DVC repository
RUN dvc init --no-scm

# configuring remote server in dvc
# RUN dvc remote add -d mylocalremote /media/sensetime/cbe421fe-1303-4821-9392-a849bfdd00e21/DVC_remote3
RUN dvc remote add -d mylocalremote /mnt/dvc_remote

RUN cat .dvc/config
# pulling the trained model
RUN dvc pull dvcfiles/trained_model.dvc


# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
