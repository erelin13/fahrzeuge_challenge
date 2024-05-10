import os
import time
import logging
import traceback
import json
import argparse
import numpy as np
from flask import Flask, request
import torch
import warnings
import random
warnings.filterwarnings('ignore')

import config
from inference_module import infer

logging.basicConfig(filename="server.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger('fahryeug_log')
logger.setLevel(logging.INFO)

app = Flask(__name__)


def convert_floatN_to_float64(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float128):
        return float(obj)
    return obj


def download_file(remote_url, savename):
    response = requests.get(remote_url)
    if response.status_code != 200:
        return None
    with open(savename, "wb") as f:
        f.write(response.content)
    return response.status_code


def preds_to_str(preds_dict):
    for k in preds_dict.keys():
        val = str(preds_dict[k])
        preds_dict[k] = val
    return preds_dict


@app.route("/ping", methods=['GET'])
def ping():
    return {"health_check": "pong"}


# if file sent in bytes, sent it with the key "raw_file"
# local file - in "local_url" key
# remotely stored file - in "remote_url" key
@app.route("/predict", methods=['POST'])
def predict():
    response = dict()
    try:
        try:
            file = request.files.to_dict()["raw_file"]
            tmp_name = file.filename
            file.save(tmp_name)
            url = tmp_name
        except:
            json_data = json.loads(request.json)
            if len(json_data["local_url"]) > 0:
                url = json_data["local_url"]
                tmp_name = None
            elif len(json_data["remote_url"]) > 0:
                tmp_name = json_data["remote_url"].replace(
                    ".", "_")
                download_file(json_data["remote_url"], tmp_name)
                url = tmp_name

        resp = infer(url)
        print(resp)
        resp = preds_to_str(resp)
        print(resp)
        if tmp_name is not None:
            os.remove(tmp_name)

        response["preds"] = resp
        response["status"] = "Success"
    except:
        response["status"] = "Unsuccessful"
    return response



if __name__ == '__main__':
    torch.set_num_threads(12)
    print("torch threas", torch.get_num_threads())
    parser = argparse.ArgumentParser(description="Prediction server")
    parser.add_argument(
        "-p", "--port", help="port for server", default=6060)
    args = vars(parser.parse_args())
    app.run(debug=True, port=args['port'], threaded=True)