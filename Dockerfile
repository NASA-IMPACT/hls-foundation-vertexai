FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./code/geospatial_fm /app/geospatial_fm
COPY ./code/setup.py /app/setup.py
RUN python /app/setup.py install

COPY requirements.txt requirements.txt

COPY ./code/ /app/

RUN pip install -r requirements.txt
