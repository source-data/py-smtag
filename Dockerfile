# docker build --ssh -t tl/smtag:multiconv .
# nvidia-docker run --shm-size 8G --rm -it -v /raid/lemberge/py-smtag:/workspace/py-smtag -p12346:6005 tl/smtag:multiconv
FROM nvcr.io/nvidia/pytorch:19.05-py3
COPY ../vsearch /workspace/vsearch
RUN pip install --upgrade pip setuptools && \
    pip install -e /workspace/py-smtag && \
    pip install tensorflow==1.8 && \
    pip install tensorboardX && \
    pip install -e git+git@github.com:source-data/vsearch.git@multihead#egg=vsearch && \
    mkdir -p /workspace/py-smtag/resources && \
    smtag-meta --help -w /workspace/py-smtag/resources \
    # smtag-predict --help -w /workspace/py-smtag/resources && \
    rm -Rf /workspace/py-smtag
