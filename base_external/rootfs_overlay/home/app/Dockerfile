# Use an official Python 3.10.8 image as the base image
FROM python:3

# Set the working directory in the container
WORKDIR /app

# # Install Python packages for face recognition
RUN apt-get update && apt-get install -y \
    python3-opencv

RUN apt-get update && apt-get install -y \
    python3-sklearn python3-sklearn-lib


# # Install gstreamer plugins to fix gstreamer cannot create pipeline error
RUN apt-get install -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad

VOLUME /data

# Define how to run your Python script
CMD ["python3.11", "/data/face_client.py"]

