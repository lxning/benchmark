FROM nvcr.io/nvidia/tritonserver:20.12-py3

RUN apt-get update \
&& curl -s http://nginx.org/keys/nginx_signing.key | apt-key add - \
&& echo 'deb http://nginx.org/packages/ubuntu/ bionic nginx' >> /etc/apt/sources.list \
&& apt-get update \
&& apt-get -y install --no-install-recommends \
    nginx \
#    nginx-module-njs \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/* \
&& pip install transformers==4.0.0

COPY --chown=1000:1000 nvidia_entrypoint.sh /opt/tritonserver
RUN chmod 0755 /opt/tritonserver/nvidia_entrypoint.sh
COPY  ngnix.conf /opt/tritonserver
