FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libxml2-dev \
    libxslt1-dev \
    libpq-dev \
    postgresql \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_py312.txt .

RUN pip install --upgrade pip setuptools
RUN pip install --no-cache-dir -r requirements_py312.txt

COPY . .
RUN pip install .

CMD ["bash"]

# (cd tmp/docker && docker run -it --rm -v "$PWD:$PWD" -w "$PWD" s2s)
# ... do your work normally inside docker shell
# exit