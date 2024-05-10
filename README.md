[EN]

### Computer Vision Challange

To solve the task on the meer base level we should at least to benchmark the easiest way, so to say, define a starting point, from which in the future we can more further, improve model and/or algorithm and comapre it with the easiest benchmark.
<br> That is why as a first solution I used a well-known, "old but gold" model architecture ResNet-50 as the main backbone (I only chnaged last layers to suit our task).
<br> In the future we could either work with a benchmark model (learning rate scheduler application, custom loss functions development etc.) or try another model.
<br> If we would like to optimize any model on inference time, we could also perform experimentations with half-presicion or model weights prunging, as well as other optimization techniques, f.i. converting model to ONNX format.

To be continued

### Instructions

If you want to test the scripts without building Docker, make sure the environment is right. Ensure anaconda or miniconda is installed on the system. Build environment as follows:
```
conda env create -f conda_env.yml
```

API can be called manually with gunicorn, f.e. like this
```
gunicorn -b 0.0.0.0:6060 --timeout 24 --max-requests 2200 --max-requests-jitter 20  --graceful-timeout 20  --keep-alive 40 "app:app"
```

Using python you can post an image path (or raw bytes - "raw_file" key, or remote image - "remote_url" key) using requests library:
```
>>> import requests
>>> import json

>>> json_paylod = json.dumps({"local_url": "/home/val/Downloads/fahrzeuge_challenge/CodingChallenge_v2/imgs/0c5720e5-baf0-4e29-b189-b4bac7dfe6e1.jpg"})
>>> requests.post("http://localhost:6060/predict", json=json_paylod)
<Response [200]>
>>> requests.post("http://localhost:6060/predict", json=json_paylod).text
'{"preds":{"perspective_score_backdoor_left":"0.83541006","perspective_score_hood":"0.9825754"},"status":"Success"}\n'
>>> 
```

[DE]

Um die Aufgabe auf der Basisebene zu lösen, sollten wir zumindest den einfachsten Weg vergleichen, also einen Ausgangspunkt definieren, von dem aus wir in Zukunft das Modell und/oder den Algorithmus weiter verbessern und mit dem einfachsten vergleichen können Benchmark.
<br> Aus diesem Grund habe ich als erste Lösung eine bekannte, „alte, aber goldene“ Modellarchitektur ResNet-50 als Haupt-Backbone verwendet (ich habe nur die letzten Schichten geändert, um sie an unsere Aufgabe anzupassen).