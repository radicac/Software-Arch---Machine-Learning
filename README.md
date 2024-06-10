First build the model.py to produce the two models linear.ml and svm.ml

> python model.py

To start the application run 

> fastapi run main.py

To build a docker image from the Dockerfile

> docker build -t diabet:0.0.2 .

To run the docker image

> docker run diabet:0.0.2
