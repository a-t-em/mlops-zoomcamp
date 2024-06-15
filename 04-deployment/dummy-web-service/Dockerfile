FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# do stuff here
WORKDIR /app 

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy #--system installs the environment in the parent OS in the container

COPY [ "starter.py" ] 

EXPOSE 9696 

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]