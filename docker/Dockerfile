FROM continuumio/miniconda3

WORKDIR /cutest

# Install dependencies
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential git gfortran python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 -p "$(openssl passwd -1 Passw0rd)" && \
    mkdir -p /app /persistent && \
    chown app:app /persistent /app

RUN git clone https://github.com/ralna/ARCHDefs ./archdefs
RUN git clone https://github.com/ralna/SIFDecode ./sifdecode
RUN git clone https://github.com/ralna/CUTEst ./cutest
RUN git clone https://bitbucket.org/optrove/sif ./mastsif

ENV ARCHDEFS /cutest/archdefs/
ENV SIFDECODE /cutest/sifdecode/
ENV MASTSIF /cutest/mastsif/
ENV CUTEST /cutest/cutest/
ENV MYARCH "pc64.lnx.gfo"

RUN wget https://raw.githubusercontent.com/jfowkes/pycutest/master/.install_cutest.sh
RUN chmod +x .install_cutest.sh
RUN ./.install_cutest.sh

# ENV PYCUTEST_CACHE /cutest/pycutest_cache
# RUN mkdir -p /cutest/pycutest_cache

USER app
WORKDIR /app

# Install python packages
ADD --chown=app:app Pipfile Pipfile.lock /app/

RUN pipenv sync

ADD --chown=app:app . /app

ENTRYPOINT tail -f /dev/null