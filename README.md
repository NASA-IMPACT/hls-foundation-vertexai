<!---- Provide an overview of what is being achieved in this repo ----> 
# HLS Foundation model in Google Cloud Platform Vertex AI

This repo deploys the recently published finetuned models based on Harmonized Landsat and Sentinel-2 (HLS) into Google Cloud Platform (GCP)'s Vertex AI. We host the models in Vertex AI's endpoint.

# Steps to deploy:

**Note: These steps are also presented in [the notebook](notebooks/IMPACT_SERVIR_HLS_FM_Customer_Container_G4G.ipynb)**

1. Clone this repository `git clone https://github.com/nasa-impact/hls-foundation-vertexai.git`
2. Change directory into the cloned repository `cd hls-foundation-vertexai`
3. [Initialize gcloud](https://cloud.google.com/sdk/docs/initializing)
4. Install required packages:
```
# Required in Docker serving container
! pip3 install -U  -r requirements.txt -q --user

# For local FastAPI development and running
! pip3 install -U  "uvicorn[standard]>=0.12.0,<0.14.0" fastapi~=0.63 -q --user

# Vertex SDK for Python
! pip3 install --upgrade --quiet  google-cloud-aiplatform --user
```
5. Check to see if gcloud configuration is done properly.
```
# List configuration
! gcloud config list

# List projects
! gcloud projects list
```
6. Use GCP code build to create and push new artifact to be used in vertex AI
```
# Replace <project-id> with one of the project ids from above.
! gcloud builds submit --region=us-central1 --tag=us-central1-docker.pkg.dev/<project-id>/hls-foundation-vertexai/inference
```
7. Register artifact as model in vertex AI (Please use notebooks/colab from here on)
```
from google.cloud import aiplatform

model = aiplatform.Model.upload(
    display_name='hls-inference',
    serving_container_image_uri="us-central1-docker.pkg.dev/<project-id>/hls-foundation-vertexai/inference",
)
```
8. Create new vertex AI endpoint
```
endpoint = model.deploy(machine_type="n1-standard-4", accelerator_type='NVIDIA_TESLA_V100', accelerator_count=1)
endpoint.to_dict()['deployedModels'][0]['id']
```
9. Create test data
```
%%writefile test.json
{
  "instances":{"date":"2023-08-13","bounding_box":[-156.81605703476012,20.69675592885614,-156.41605703476014,21.096755928856137], "model_id": "burn_scars"},
}

```
10. Test endpoint

**Note: Replace all <PROJECT_ID> with your project id, and <ENDPOINT_ID> with the output from step 8.**

```
# Get inference from the deployed endpoint. Copy over the endpoint id from above and replace <ENDPOINT_ID>, and project id from about to replace <PROJECT_ID>
! export ENDPOINT_ID=<ENDPOINT_ID>; export PROJECT_ID=<PROJECT_ID>; export INPUT_DATA_FILE="test.json"; curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/endpoints/${ENDPOINT_ID}:predict \
-d "@${INPUT_DATA_FILE}"
```
