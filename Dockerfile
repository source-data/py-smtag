# nvidia-docker run --shm-size 8G --rm -it -v /raid/lemberge/py-smtag:/workspace/py-smtag -p12346:6005 smtag:dev
FROM nvcr.io/nvidia/pytorch:19.05-py3
COPY . /workspace/py-smtag

RUN pip install --upgrade pip setuptools && \
    pip install -r /workspace/py-smtag/requirements.txt && \
    rm -Rf /workspace/py-smtag

