#!/bin/bash

cd ../
# Execute the command inside the Apptainer container
apptainer exec \
  --nv \
  --env-file .env \
  --bind $(pwd)/:/app/ \
  --pwd /app \
  generativezoo.sif /bin/bash scripts/main.sh