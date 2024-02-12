FROM tiangolo/uwsgi-nginx-flask:latest
WORKDIR /app
COPY ./ /app/
RUN pip3 install -r requirements.txt