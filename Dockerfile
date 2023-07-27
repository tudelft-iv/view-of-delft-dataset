# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Install git
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

# Clone the git repository
RUN git clone https://github.com/tudelft-iv/view-of-delft-dataset.git .

# Install any necessary dependencies
RUN pip install numpy opencv-python-headless pandas sklearn seaborn

# Run evaluation_script.py when the container launches
ENTRYPOINT ["python", "./evaluation_script.py"]
