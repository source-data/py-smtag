# syntax=docker/dockerfile:1.0.0-experimental
# export DOCKER_BUILDKIT=1
# docker build -t tl/smtag:multiconv .
# nvidia-docker run --shm-size 8G --rm -it -v /raid/lemberge/py-smtag:/workspace/py-smtag -p12346:6005 tl/smtag:multiconv
FROM nvcr.io/nvidia/pytorch:19.05-py3
COPY . /workspace/py-smtag

# http://blog.oddbit.com/post/2019-02-24-docker-build-learns-about-secr/
# This is necessary to prevent the "git clone" operation from failing
# with an "unknown host key" error.
RUN mkdir -m 700 /root/.ssh; \
  touch -m 600 /root/.ssh/known_hosts; \
  ssh-keyscan github.com > /root/.ssh/known_hosts

# This command will have access to the forwarded agent (if one is
# available)
# RUN --mount=type=ssh git clone --branch multihead git@github.com:source-data/vsearch
RUN pip install --upgrade pip setuptools
#RUN --mount=type=ssh pip install -e git+git@github.com:source-data/vsearch.git@multihead#egg=vsearch
RUN --mount=type=ssh git clone --branch multihead git@github.com:source-data/vsearch
RUN pip install -e /workspace/vsearch
RUN pip install -e /workspace/py-smtag
RUN pip install tensorflow==1.8
RUN pip install tensorboardX==1.6
RUN vs
RUN smtag-meta --help -w /workspace/py-smtag/resources
    # smtag-predict --help -w /workspace/py-smtag/resources && \
RUN rm -Rf /workspace/py-smtag
RUN rm -Rf /workspace/vsearch
