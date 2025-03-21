
docker run --rm --gpus all --ipc=host --env-file ../.env \
 -v $(pwd)/../.secrets:/app/.secrets \
 -v $(pwd)/../data:/app/data \
 -v $(pwd)/../src:/app/src \
 -v $(pwd)/../models:/app/models \
 -v $(pwd)/../scripts:/app/scripts \
 generativezoo /bin/bash scripts/main.sh