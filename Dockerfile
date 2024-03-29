FROM centos:7.2.1511

# author label
LABEL maintainer="jclian"

# install related packages
ENV ENVIRONMENT DOCKER_PROD
RUN cd / && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && yum makecache \
    && yum install -y wget aclocal automake autoconf make gcc gcc-c++ python-devel mysql-devel bzip2 libffi-devel epel-release \
    && yum clean all

# install python 3.7.0
RUN wget https://npm.taobao.org/mirrors/python/3.7.0/Python-3.7.0.tar.xz \
    && tar -xvf Python-3.7.0.tar.xz -C /usr/local/ \
    && rm -rf Python-3.7.0.tar.xz \
    && cd /usr/local/Python-3.7.0 \
    && ./configure && make && make install

# install related packages
RUN yum install -y python-pip \
    && yum install -y python-setuptools \
    && mkdir -m 755 -p /etc/supervisor/conf.d \
    && yum install -y supervisor \
    && pip3 install --upgrade pip -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com \
    && pip3 install setuptools==33.1.1 -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com \
    && yum clean all

# expost port
EXPOSE 15731
