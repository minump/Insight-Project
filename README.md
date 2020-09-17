# Insight-Project

# Smartly Financial : A scalable credit lending web application.
Minu Mathew, Insight AI Fellowship (AISV 2020B) Project. 
This is a consulting project for [RedCarpetUp](https://www.redcarpetup.com)

## Motivation :
Banks use credit scores to determine if you're trustworthy or not. However, more than 2.5 billion people around the world do not have one, to begin with. Just because of some financial paperwork, the world is missing out on new entrepreneurs, ideators, and innovators. This is especially significant in the present situation of COVID Relief funds. The red tape involved with banks caused weeks of delay for COVID Relief funds to be dispersed. 
We need an alternative form of risk assessment for lenders. This needs to be explainable, given GDPR and other compliance issues in the financial sector, and there should not be any black boxes, if a decision has been made, we need to know why that decision has been made.

Smartly Financial is an internal tool for scalable and explainable lending. It uses a more holistic view of user mobile data points to create a financial identity. It uses XGBoost model due to its superior performance in handling large and extremely imbalanced datasets and SHAP values for explainability. Smartly Financial is built as a docker container and is hosted on the cloud via GCP. The product is scalable as it is deployed using the Google Kubernetes engine and has a load balancer that can handle multiple server requests. A Streamlit frontend serves as a dashboard to display model performances.

## Demo:
Smartly Financial web app is available [here](http://34.105.46.31/).
Presentation slides are available [here](https://docs.google.com/presentation/d/1Nr9nRDg-IQUO-4ehwH3oP430ftY2rCf1ZdMvc7Hcq0w/edit#slide=id.p1).

## Files :
- data
  - The project utilized the data from RedCarpet. This data cannot be disclosed as its proprietary data. The model can be tested out for Kaggle [Home credit application loan dataset](https://www.kaggle.com/c/home-credit-default-risk/data?select=application_train.csv), which has a similar feature structure to the RedCarpet data.
- model
  - LogisticRegression.py trains and predicts using Logistic Regression model
  - XGBoost.py train and predicts using XGBoost Classifier. This is used as the default model because of this higher performance scores.
  - trained models are saved in H5 format in "saved models" folder.
- explain
  - Shap.py uses SHAP values to explain the feature importance and feature dependancies. SHAP plots are saved in this folder
- utlis
  - read_data.py unzips the data file and reads into csv file.
  - data_processing.py takes care of the data preprocessing steps
  - df_one_hot_encoding.py encodes categorical variables into one-hot-encoded features.
  - sampling.py samples the class imbalanced dataset according to the user defined sampling method.
  - split_data.py splits the data into train, validation and test sets.
- app.py is the main file to run.
- Dockerfile
- requirements.txt

## Please Note:
The model has been tuned for RedCarpet data. Hence the results on Kaggle home credit application loan dataset might not be up-to-the-mark.
## Results :
Data : RedCarpet\
Classes:\
Class 0 - Paid Class (majority class)\
Class 1 - Defaulted Class (minority class)

| Metrics          | Logistic Regression | XGBoost Classifier|
| -------------    |:-------------------:|:-----------------:|
| Class 0 F1 score | 0.926               |   0.995           |
| Class 1 F1 score | 0.963               |   0.989           |
| False positives  | 108                 |     14            |
| False negatives	 | 74                  | 11                |

## Installation :
1. Clone the GitHub repository
```
git clone https://github.com/minump/Insight-Project/
```
2. Change working directory
```
cd Insight-Project
```
3. Install dependencies: The model is tested on Python 3.8, with dependencies listed in requirements.txt. To install these Python dependencies, please run
```
pip install -r requirements.txt
```
Or if you prefer to use conda,
```
conda install --file requirements.txt
```
## Usage :
1. To request all command line parameters, please run:
```
python app.py --help
```
2. To run Smartly Financial python module with default parameters, please run:
```
python app.py
```
3. To run Smartly Financial streamlit app in your local, please run:
```
streamlit run app.py
```
and point your browser to http://0.0.0.0:8501 .\
4. To run the containarized web application
- Download and install [Docker](https://docs.docker.com/get-docker/) to create a containerized application for the demo.
- Run
```
docker build -t smartlyfinancial-streamlit:v1 -f Dockerfile .
```
- Check for the docker image by
```
docker images
```
- Run the dockerized app locally by
```
docker run -p 8501:8501 smartlyfinancial-streamlit:v1
```
5. To see the web app deployed in GKE go [here](http://34.105.46.31/)

## Run in GCP using Kubernetes :
### Prerequisites
1. [Google Cloud SDK](https://cloud.google.com/sdk/install)
2. Kubenetes SDK, run the following command to install
```
gcloud components install kubectl
```
3. [Docker](https://docs.docker.com/get-docker/).
4. GCP account with billing enabled. GCP project ID {PROJECT_ID} and compute_zone.

### Workflow

1. Dockerize the app
```
docker build -t gcr.io/{$PROJECT_ID}/smartlyfinancial-streamlit:v1 .
```
2. Test the container locally
```
docker run --rm -p 8501:8501 gcr.io/{$PROJECT_ID}/smartlyfinancial-streamlit:v1
```
Then point your internet browser to localhost:8501 to see the app.\
3. Push image to Google Container Registry (GCR)
```
gcloud auth configure-docker
docker push gcr.io/{$PROJECT_ID}/smartlyfinancial-streamlit:v1
```
4. Create a container cluster (in this example with 2 nodes). If you have already created a cluster with the gcloud container clusters, only the last step is necessary.
```
gcloud config set project {$PROJECT_ID}
gcloud config set compute/zone {$COMPUTE_ZONE}
gcloud container clusters create {$CLUSTER_NAME} --num-nodes=2
gcloud container clusters get-credentials {$CLUSTER_NAME} --zone {$COMPUTE_ZONE} --project {$PROJECT_ID}
```
5. Deploy your application: Make multiple copies of your app via Kubernetes clusters management system.
```
kubectl create deployment smartlyfinancial-webapp --image=gcr.io/{$PROJECT_ID}/smartlyfinancial-streamlit:v1
```
6. Expose your application to the internet: Assign external IP addresses for your app so it is accessible from the internet.
```
kubectl expose deployment smartlyfinancial-webapp --type=LoadBalancer --port 80 --target-port 8501
```
7. Find the service using
```
kubectl get service
```
Point your browser to the External URL to access the app.

