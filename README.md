# TF-serving

## TensorFlow Serving with Docker
TensorFlow Serving is a flexible, high-performance serving system for machine learning models, designed for production environments. It deals with the inference aspect of machine learning, taking models after training and managing their lifetimes, providing clients with versioned access via a high-performance, reference-counted lookup table. TensorFlow Serving provides out-of-the-box integration with TensorFlow models, but can be easily extended to serve other types of models and data.

#### To note a few features:
Can serve multiple models, or multiple versions of the same model simultaneously
Exposes both gRPC as well as HTTP inference endpoints
Allows deployment of new model versions without changing any client code
Supports canarying new versions and A/B testing experimental models
Adds minimal latency to inference time due to efficient, low-overhead implementation
Features a scheduler that groups individual inference requests into batches for joint execution on GPU, with configurable latency controls
Supports many servables: Tensorflow models, embeddings, vocabularies, feature transformations and even non-Tensorflow-based machine learning models

## Build it from Scratch
The below mentioned steps need to be followed after exporting the tensorflow model to desired location. The location used in the below commands is ‘./saved_model/1’.

#### Step1: Download the TensorFlow Serving Docker image
For CPU serving:
docker pull tensorflow/serving 

For GPU serving:
docker pull tensorflow/serving:latest-gpu  (For GPU)


#### Step2: Start TensorFlow Serving container and open the REST API port
For CPU:

docker run -p 8527:8527 \
--mount type=bind,source=/home/jupyter/saved_model,target=/models/embeddings \
-e MODEL_NAME=embeddings -t tensorflow/serving &

docker run -p 8501:8501 --name tfserving_resnet  --mount type=bind,source=/home/jupyter/saved_model,target=/models/resnet  -e MODEL_NAME=resnet -t tensorflow/serving &

For GPU:
docker run --gpus all -p 8501:8501 --name tfserving_resnet  --mount type=bind,source=/home/jupyter/saved_model, target=/models/resnet  -e MODEL_NAME=resnet -t tensorflow/serving:latest-gpu &

## Breaking down the command line arguments:
-p 8501:8501 : Publishing the container’s port 8501 (where TF Serving responds to REST API requests) to the host’s port 8501

--name tfserving_resnet : Giving the container we are creating the name “tfserving_resnet” so we can refer to it later

--mount type=bind, source=/home/jupyter/saved_model, target=/models/resnet : Mounting the host’s local directory (/home/jupyter/saved_model) on the container (/models/resnet) so TF Serving can read the model from inside the container.\

-e MODEL_NAME=resnet : Telling TensorFlow Serving to load the model named “resnet”

-t tensorflow/serving / -t tensorflow/serving:latest-gpu  : Running a Docker container based on the serving image “tensorflow/serving”/”tensorflow/serving:latest-gpu” 
Note-: Change the source and target  variable names as per the requirement. 

#### Step3: Query the model using this command
python3 Resnet_client.py --url 'http://localhost:8501/v1/models/resnet:predict' --image_path 'user001.jpg'

Returns =>  "Embeddings": [0.34514, .024568, …….,0.55932] 

This Resnet_client.py script will provide the embeddings of the image. It contains the Predict API with the image preprocessing function.

#### Step4: Kill the tensorflow serving container (optional)
docker kill tfserving_resnet


Now that we have TensorFlow Serving running with Docker, we can deploy our machine learning models in containers easily while maximizing ease of deployment and performance.
