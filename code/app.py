from fastapi import FastAPI, Request

import joblib
import json
import numpy as np
import pickle
import os

from google.cloud import storage
from huggingface_hub import huggingface_hub_download
from preprocess import MySimpleScaler
from sklearn.datasets import load_iris


app = FastAPI()
gcs_client = storage.Client()

with open('burnscars.pth', 'wb') as burnscars_file, open('flood.pth', 'wb') as flood_file:
    gcs_client.download_blob_to_file(
        f"{os.environ['AIP_STORAGE_URI']}/preprocessor.pkl", preprocessor_f
    )
    gcs_client.download_blob_to_file(f"{os.environ['AIP_STORAGE_URI']}/model.joblib", model_f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

_class_names = load_iris().target_names
_model = joblib.load('model.joblib')
_preprocessor = preprocessor


@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {}


@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    body = await request.json()

    instances = body['instances']
    inputs = np.asarray(instances)
    preprocessed_inputs = _preprocessor.preprocess(inputs)
    outputs = _model.predict(preprocessed_inputs)

    return {'predictions': [_class_names[class_num] for class_num in outputs]}


import os
import boto3
import rasterio

from app.lib.model import Infer
from app.lib.downloader import Downloader
from app.lib.post_process import PostProcess
from typing import Union

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BUCKET_NAME = '2023-igarss-tutorial-store'

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

MODEL_CONFIGS = {
    'flood': {
        'config': 'sen1floods11_Prithvi_100M.py',
        'repo': 'ibm-nasa-geospatial/Prithvi-100M-sen1floods11',
        'weight': 'sen1floods11_Prithvi_100M.pth',
        'collections': ['HLSS30'],
    },
    'burn_scars': {
        'config': 'burn_scars_Prithvi_100M.py',
        'repo': 'ibm-nasa-geospatial/Prithvi-100M-burn-scar',
        'weight': 'burn_scars_Prithvi_100M.pth',
        'collections': ['HLSS30', 'HLSL30'],
    },
    'crop_classification': {
        'config': 'multi_temporal_crop_classification_Prithvi_100M.py',
        'repo': 'ibm-nasa-geospatial/Prithvi-100M-multi-temporal-crop-classification',
        'weight': 'multi_temporal_crop_classification_Prithvi_100M.pth',
        'collections': ['HLSS30', 'HLSL30'],
    },
}

MODELS = {model_name: load_model(model_name) for model_name in MODEL_CONFIGS}


def load_model(model_name):
    repo = MODEL_CONFIGS[model_name]['repo']
    config = huggingface_hub_download(repo, filename=MODEL_CONFIGS[model_name]['config'])
    model_path = huggingface_hub_download(repo, filename=MODEL_CONFIGS[model_name]['weight'])
    infer = Infer(config, model_path)
    _ = infer.load_model()
    return infer


def assumed_role_session():
    client = boto3.client('sts')
    creds = client.assume_role(RoleArn=ROLE_ARN, RoleSessionName=ROLE_NAME)['Credentials']
    return boto3.session.Session(
        aws_access_key_id=creds['AccessKeyId'],
        aws_secret_access_key=creds['SecretAccessKey'],
        aws_session_token=creds['SessionToken'],
        region_name='us-east-1',
    )


def download_file(s3_link, file_name):
    session = assumed_role_session()
    s3 = session.client('s3')
    object_name = s3_link.replace('s3://', '')
    s3.download_file(BUCKET_NAME, object_name, file_name)


@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {'Hello': 'World'}


@app.get('/models')
def list_models():
    response = jsonable_encoder(list(MODEL_CONFIGS.keys()))
    return JSONResponse({'models': response})


@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def infer_from_model(request: Request):
    body = await request.json()

    instances = body['instances']
    model_id = instances['model_id']
    infer_date = instances['date']
    bounding_box = instances['bounding_box']

    if model_id not in MODELS:
        response = {'statusCode': 422}
        return JSONResponse(content=jsonable_encoder(response))
    infer = MODELS[model_id]
    all_tiles = list()
    geojson_list = list()

    for layer in MODEL_CONFIGS[model_id]['collections']:
        downloader = Downloader(infer_date, layer)
        all_tiles += downloader.download_tiles(bounding_box)
    if all_tiles:
        results = infer.infer(all_tiles)
        transforms = list()
        for tile in all_tiles:
            with rasterio.open(tile) as raster:
                transforms.append(raster.transform)
        for index, result in enumerate(results):
            detections = PostProcess.extract_shapes(result, transforms[index])
            detections = PostProcess.remove_intersections(detections)
            geojson = PostProcess.convert_to_geojson(detections)
            for geometry in geojson:
                updated_geometry = PostProcess.convert_geojson(geometry)
                geojson_list.append(updated_geometry)
    del infer
    gc.collect()
    torch.cuda.empty_cache()
    final_geojson = {'predictions': {'type': 'FeatureCollection', 'features': geojson_list}}
    return JSONResponse(content=jsonable_encoder(final_geojson))
