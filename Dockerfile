# Start from a core stack version
FROM continuumio/miniconda3

# Set the working directory in the container to /app
WORKDIR /app

# Install git
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

# Clone the git repository from the eval-docker branch
RUN git clone -b eval-docker https://github.com/tudelft-iv/view-of-delft-dataset.git .

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Ensure the environment is activated:
RUN echo "Make sure the environment is activated: "
RUN echo $CONDA_DEFAULT_ENV

# Run evaluation_script.py when the container launches
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "./evaluation_script.py"]
