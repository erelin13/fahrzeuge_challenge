[EN]

### Computer Vision Challange

To solve the task on the meer base level we should at least to benchmark the easiest way, so to say, define a starting point, from which in the future we can more further, improve model and/or algorithm and comapre it with the easiest benchmark.
<br> That is why as a first solution I used a well-known, "old but gold" model architecture ResNet-50 as the main backbone (I only chnaged last layers to suit our task). I did not chase the best results and top scores here, that could take a long time.
<br> However, in the future we could either work with a benchmark model (learning rate scheduler application, custom loss functions development etc.) or try another model.
<br> If we would like to optimize any model on inference time, we could also perform experimentations with half-presicion or model weights prunging, as well as other optimization techniques, f.i. converting model to ONNX format.

### Instructions

If you want to test the scripts without building Docker, make sure the environment is right. Ensure anaconda or miniconda is installed on the system. Build environment as follows:
```
conda env create -f conda_env.yml
```
[Download model weight](https://drive.google.com/file/d/12-odTNp8RvMIDMUhupyAwKLTgRZXmHa4/view?usp=sharing) and place them in the root directory. If you want to keep them elsewehre, make sure you specify `model_checkpoint` path in config.py file.

API can be called manually with gunicorn, f.e. like this
```
gunicorn -b 0.0.0.0:6060 --timeout 24 --max-requests 2200 --max-requests-jitter 20  --graceful-timeout 20  --keep-alive 40 "app:app"
```
or even more simplier
```
python app.py -p 6060
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
Finally, if you want to build Docker image and run application in container, you can build an image quiet simply:
```
[sudo] docker build .
```
Then run container. I show only a localhost mapping, adjust as needed.
```
[sudo] docker run -dp 0.0.0.0:6060:6060 <IMAGE_ID>
```

[DE]

Um die Aufgabe auf der Basisebene zu lösen, sollten wir zumindest den einfachsten Weg vergleichen, also einen Ausgangspunkt definieren, von dem aus wir in Zukunft das Modell und/oder den Algorithmus weiter verbessern und mit dem einfachsten vergleichen können Benchmark.
<br> Aus diesem Grund habe ich als erste Lösung eine bekannte, „alte, aber goldene“ Modellarchitektur ResNet-50 als Haupt-Backbone verwendet (ich habe nur die letzten Schichten geändert, um sie an unsere Aufgabe anzupassen). Ich bin hier nicht auf die Jagd nach den besten Ergebnissen und Bestnoten gegangen, das kann lange dauern.
<br> In Zukunft könnten wir jedoch entweder mit einem Benchmark-Modell arbeiten (Learning Rate Scheduler-Anwendung, Entwicklung benutzerdefinierter Verlustfunktionen usw.) oder ein anderes Modell ausprobieren.
<br> Wenn wir ein Modell hinsichtlich der Inferenzzeit optimieren möchten, könnten wir auch Experimente mit Halbgenauigkeit oder Modellgewichtungsreduzierung sowie anderen Optimierungstechniken durchführen, z. B. Konvertieren des Modells in das ONNX-Format.

### Anweisungen

Wenn Sie die Skripte testen möchten, ohne Docker zu erstellen, stellen Sie sicher, dass das Environment richtig ist. Stellen Sie sicher, dass Anaconda oder Miniconda auf dem System installiert ist. Erstellen Sie das Environment wie folgt:
```
conda env create -f conda_env.yml
```
[Herunterladen Sie Modellgewichte](https://drive.google.com/file/d/12-odTNp8RvMIDMUhupyAwKLTgRZXmHa4/view?usp=sharing) und legen Sie sie im Stammverzeichnis ab. Wenn Sie sie an einem anderen Ort behalten möchten, stellen Sie sicher, dass Sie „model_checkpoint“ in der Datei config.py angeben.

API kann manuell mit gunicorn aufgerufen werden, z.B. so was
```
gunicorn -b 0.0.0.0:6060 --timeout 24 --max-requests 2200 --max-requests-jitter 20  --graceful-timeout 20  --keep-alive 40 "app:app"
```
oder sogar einfacher
```
python app.py -p 6060
```

Mit Python können Sie einen Bildpfad (oder Rohbytes – Schlüssel "raw_file" oder Remote-Bild – Schlüssel "remote_url") mithilfe der Anforderungsbibliothek veröffentlichen:
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
Endlich können Sie auch Docker image erstellen und Anwendung im Container ausführen. Einfach starten Sie so was
```
[sudo] docker build .
```
Danach führen Sie den Container aus, z.B so was
```
[sudo] docker run -dp 0.0.0.0:6060:6060 <IMAGE_ID>
```