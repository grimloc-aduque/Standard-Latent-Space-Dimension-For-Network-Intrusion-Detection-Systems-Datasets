#Deriving the latest base image
FROM tensorflow/tensorflow:latest

WORKDIR /usr/src

COPY ./requirements.txt ./
COPY ./f01_preprocess_datasets.py ./
COPY ./f02_autoencoders.py ./
COPY ./f03_encoded_classification.py ./
COPY ./f04_encoded_datasets.py ./

COPY ./datasets_clean ./datasets_clean
COPY ./requirements.txt ./
RUN mkdir ./X_encoded

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD [ "python3", "./f04_encoded_datasets.py"]
