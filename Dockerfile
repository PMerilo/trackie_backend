FROM python:3.11 

WORKDIR /app
COPY ./ /app/

RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade tensorflow
RUN pip3 install --upgrade keras
RUN apt update
RUN apt-get install -y ffmpeg