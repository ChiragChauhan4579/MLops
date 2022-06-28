FROM continuumio/miniconda3:4.9.2
RUN pip install mlflow==1.18.0 protobuf==3.19.0
COPY ./training_project /src
WORKDIR /src
ENTRYPOINT [ "mlflow","run","." ]