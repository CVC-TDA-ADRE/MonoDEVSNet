ARG IMAGE_BASE

FROM ${IMAGE_BASE}

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
        python3-opencv python3.6 python3.6-dev git wget sudo python3-distutils

RUN wget https://bootstrap.pypa.io/get-pip.py && \
        python3.6 get-pip.py && \
        rm get-pip.py

COPY requirements.txt /software/requirements.txt
RUN pip3.6 install -r /software/requirements.txt

COPY ./entrypoint_dev.sh /entrypoint_dev.sh
RUN chmod +x /entrypoint_dev.sh
ENTRYPOINT ["/entrypoint_dev.sh"]
CMD ["bash"]
WORKDIR /
