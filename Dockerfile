# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
# COPY requirements.txt .
COPY requirements_part1.txt .
COPY requirements_part2.txt .

# Install the dependencies and increase timeout
# RUN pip install --default-timeout=300 -r requirements_part1.txt
RUN pip install --default-timeout=300 -r requirements_part1.txt --no-deps
RUN pip install scipy==1.11.4
RUN pip install --default-timeout=300 -r requirements_part2.txt

# RUN pip install --no-cache-dir -r requirements.txt --timeout=120
# RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set environment variables
ENV FLASK_APP=run.py

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["flask", "run", "--host=0.0.0.0"]
