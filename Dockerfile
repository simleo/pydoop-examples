ARG HADOOP_MAJOR_VERSION=3
FROM crs4/pydoop-base:${HADOOP_MAJOR_VERSION}
MAINTAINER simone.leo@crs4.it

COPY examples /examples
WORKDIR /examples

RUN source /etc/profile && \
    yum install python36u-tkinter && \
    pip3 install --no-cache-dir --upgrade -r requirements.txt
