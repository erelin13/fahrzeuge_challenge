#!/bin/bash

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

if conda info --envs | grep -q challange; then echo "challange environment already exists"; else conda env create -f conda_env.yml; fi
conda init
conda activate challange

__PORT=$1

echo "Port in use: $__PORT"

# alias gunicorn=$(conda info --base)/envs/challange/bin/gunicorn

gunicorn -b 0.0.0.0:$__PORT --timeout 24 --max-requests 2200 --max-requests-jitter 20  --graceful-timeout 20  --keep-alive 40 "app:app"