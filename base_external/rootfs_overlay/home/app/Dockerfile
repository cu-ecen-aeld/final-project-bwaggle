# Use an official Python 3.10.8 image as the base image
FROM python:3.10.8-slim

# Set the working directory in the container
WORKDIR /app

# Install Python packages for face recognition
RUN apt-get update && apt-get install -y \
    python3-opencv \
    python3-sklearn \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*


# This can be removed
RUN apt-get update && apt-get install -y xdg-utils

# Install gstreamer plugins to fix gstreamer cannot create pipeline error
RUN apt-get install -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad

# Copy your Python script to the container
#COPY face.py .

VOLUME /data

# Install scikit-learn, OpenCV, and any other dependencies
#RUN pip install scikit-learn opencv-python

# Define how to run your Python script
CMD ["python3.9", "/data/face_app.py"]

