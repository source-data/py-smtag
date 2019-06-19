FROM nvcr.io/nvidia/pytorch:19.05-py3
COPY . /workspace/py-smtag
RUN pip install --upgrade pip setuptools && \
    pip install -e /workspace/py-smtag && \
    pip install tensorflow==1.8 && \
    pip install tensorboardX && \
    mkdir -p /workspace/py-smtag/resources && \
    smtag-meta --help -w /workspace/py-smtag/resources \
    # smtag-predict --help -w /workspace/py-smtag/resources && \
    rm -Rf /workspace/py-smtag
