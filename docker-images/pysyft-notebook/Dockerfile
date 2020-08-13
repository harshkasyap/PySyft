FROM continuumio/miniconda3

ENV WORKSPACE /workspace

# Setup workspace environment
RUN apt-get update && apt-get install -y gcc && apt-get install -y nano && apt-get install -y iputils-ping
RUN conda install jupyter notebook=5.7.8
RUN pip install --no-cache-dir syft[udacity,sandbox]
RUN pip install tensorflow-tensorboard==1.5.1
RUN pip install tensorboard==1.14.0

# Create jupyter notebook workspace
RUN mkdir $WORKSPACE
WORKDIR $WORKSPACE

# Make the image start the jupyer notebook
COPY ./entrypoint.sh /
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
