# Use a lightweight mambaforge base image
FROM condaforge/mambaforge:latest

# Set the working directory
WORKDIR /app

# Create the environment using mamba
# We combine the create and install steps to keep the image layers clean
RUN mamba create -n asx-tracker python=3.10 -y && \
    mamba install -n asx-tracker -c conda-forge pandas numpy openpyxl matplotlib plotly -y && \
    mamba clean --all -f -y

# Install pip dependencies inside the env
# We use the full path to the env's pip to ensure it lands in the right place
RUN /opt/conda/envs/asx-tracker/bin/pip install --root-user-action yfinance colorlog

# Ensure the environment is on the PATH
ENV PATH /opt/conda/envs/asx-tracker/bin:$PATH

# Copy your source code (assuming you have a main.py)
COPY . .

# Set the default command to run your script
CMD ["python", "portfolio_tracker.py"]