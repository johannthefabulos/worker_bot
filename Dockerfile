# Use the official Python 3.12.4 image from DockerHub
FROM python:3.10.11-alpine

# Set the working directory in the container
WORKDIR /live_worker

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any dependencies listed in requirements.txt
RUN apk add --no-cache --update \
    python3 python3-dev gcc \
    gfortran musl-dev
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .



# Expose port 5001 (or whichever port your app will run on)
EXPOSE 5001

# Run your application (replace 'app.py' with your entry point script)
ENTRYPOINT ["python", "-u", "main.py"]
