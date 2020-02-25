# syntax=docker/dockerfile:1.0.0-experimental
# DOCKER_BUILDKIT=1 docker build --ssh default -t tl/smtag:multiconv .
# nvidia-docker run --shm-size 8G --rm -it -v /raid/lemberge/py-smtag:/workspace/py-smtag -p12346:6005 tl/smtag:multiconv
FROM nvcr.io/nvidia/pytorch:19.05-py3
COPY . /workspace/py-smtag

# http://blog.oddbit.com/post/2019-02-24-docker-build-learns-about-secr/
# This is necessary to prevent the "git clone" operation from failing
# with an "unknown host key" error.
# RUN mkdir -m 700 /root/.ssh; \
#   touch -m 600 /root/.ssh/known_hosts; \
#   ssh-keyscan github.com > /root/.ssh/known_hosts

# This command will have access to the forwarded agent (if one is
# available)
#RUN --mount=type=ssh git clone --branch multihead git@github.com:source-data/vsearch
RUN pip install --upgrade pip setuptools && \
    pip install -e /workspace/py-smtag && \
    pip install -r /workspace/py-smtag/smtag/requirements.txt && \
    pip install tensorflow==1.8.0 && \
    pip install tensorboardX==1.6 && \
    # smtag-meta --help -w /workspace/py-smtag/resources && \
    rm -Rf /workspace/py-smtag

