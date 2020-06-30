# Dockerfile 

# pull python image
FROM python:3

# copy local files into container
COPY app.py /tmp/
COPY requirements.txt /tmp/
COPY model /tmp/model
COPY data /tmp/data
COPY utils /tmp/utils
COPY explain /tmp/explain

# .streamlit for something to do with making enableCORS=False
#COPY .streamlit /tmp/.streamlit 

ENV PORT 8080

# change directory
WORKDIR /tmp

# install dependencies

RUN pip install -r requirements.txt

# run commands
CMD ["streamlit", "run", "app.py"]
