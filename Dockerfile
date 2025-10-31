# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /code

# 3. Copy the requirements file and install dependencies
# We copy *only* requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code
# This includes main.py, train.py, and the housing_model.joblib
# (Though train.py isn't needed to *run*, it's simpler to copy all)
COPY . .

# 5. [IMPORTANT] Ensure the model is built if it doesn't exist.
# This runs 'python train.py' *inside the container* during the build
# if the model file isn't found.
# A better way is to build the model locally and COPY it,
# but this is a good fallback.
RUN if [ ! -f housing_model.joblib ]; then python train.py; fi

# 6. Expose the port the app will run on
# The PORT env var will be set by the hosting platform (e.g., Heroku, Cloud Run)
# We default to 8000 if it's not set.
EXPOSE ${PORT:-8000}

# 7. Define the command to run your app using Gunicorn
# Gunicorn is a production-grade server.
# -w 4: Use 4 worker processes
# -k uvicorn.workers.UvicornWorker: Use Uvicorn to handle requests
# main:app: Look in the 'main.py' file for the variable 'app'
# --bind 0.0.0.0:${PORT:-8000}: Bind to all IPs on the specified port
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:${PORT:-8000}"]
