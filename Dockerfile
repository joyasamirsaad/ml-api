FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-dev python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
WORKDIR /app
COPY requirements.txt .
RUN pip install -vvv --progress-bar=on --no-color -r requirements.txt
RUN pip install --no-cache-dir torch==2.3.1+cu121 torchvision==0.18.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir pytorch-lightning==2.2.5
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY src/ ./src
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
