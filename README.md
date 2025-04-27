# Image Classifier
This project showcases an example of utilising Flask to serve a simple webpage that interacts with an existing Keras model for image classification. 

This is the codebase that was created as part of this tutorial here:

[![Youtube video link](http://img.youtube.com/vi/0nr6TPKlrN0/0.jpg)](http://www.youtube.com/watch?v=0nr6TPKlrN0)

## Prerequisites
Install the dependencies 
``` bash
pip install -r requirements.txt
```

> **Note:** This project requires Python 3.8 or higher 

## Usage
1. To run locally either run from the IDE or use the following command:
```bash
python app.py
```

2. Open your web browser and visit `http://localhost:3000`
3. Select an image and click on the `Predict Image` button

## Licence 
This project is licensed under the MIT Licence.
```
deploy_model_foodia
├─ Dockerfile
├─ LICENSE
├─ README.md
├─ app.py
├─ bilstm_model2.h5
├─ images
│  ├─ cat2.jpg
│  ├─ dog.jpg
│  └─ receipt (3).jpg
├─ img_to_txt_ocr.py
├─ model.py
├─ predicted_results
│  └─ hasil_receipt (3).jpg.csv
├─ requirements.txt
├─ save_label_encoder2.pkl
├─ save_tokenizer2.pkl
└─ templates
   └─ index.html

```