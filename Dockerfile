FROM nvcr.io/nvidia/pytorch:19.05-py3
COPY . /workspace/py-smtag
RUN pip install --upgrade pip setuptools && \
    pip install -e /workspace/py-smtag && \
    mkdir -p resources && \
    smtag-predict --help && \
    rm -Rf /workspace/py-smtag
