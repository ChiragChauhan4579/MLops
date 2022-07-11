FROM continuumio/miniconda3:4.9.2
RUN pip install mlflow==1.18.0 protobuf==3.19.0
COPY ./training_project /src
WORKDIR /src
# ENV MLFLOW_TRACKING_URI=http://localhost:5000
ENTRYPOINT [ "mlflow","run","." ]