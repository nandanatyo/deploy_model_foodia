from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import logging
import tempfile
import shutil
import time
from img_to_txt_ocr import prepro
from model import model

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
os.makedirs("predicted_results", exist_ok=True)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK", "timestamp": time.time()})


@app.route('/api/predict', methods=['POST'])
def predict_api():
    if 'image' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    imagefile = request.files['image']
    if not imagefile.filename:
        return jsonify({"error": "No file selected for uploading"}), 400

    temp_dir = tempfile.mkdtemp()

    try:
        temp_image_path = os.path.join(temp_dir, imagefile.filename)
        imagefile.save(temp_image_path)
        app.logger.info(f"File saved temporarily at {temp_image_path}")

        app.logger.info("Processing image...")
        df_test_tokens, X_test = prepro(temp_image_path)
        app.logger.info("Running prediction model...")
        hasil_bilstm_model = model(df_test_tokens, X_test)

        output_filename = f"hasil_{
            os.path.splitext(imagefile.filename)[0]}.csv"
        output_path = os.path.join("predicted_results", output_filename)
        hasil_bilstm_model.to_csv(output_path, index=False)
        app.logger.info(f"Results saved to {output_path}")

        items = hasil_bilstm_model.to_dict('records')

        return jsonify({
            "status": "success",
            "receipt_name": imagefile.filename,
            "predictions": items
        })

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        app.logger.info(f"Cleaning up temporary directory {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

        if os.path.exists('processed_test'):
            shutil.rmtree('processed_test', ignore_errors=True)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict_web():
    if 'imagefile' not in request.files:
        return render_template('index.html', error="No file uploaded")

    imagefile = request.files['imagefile']
    if not imagefile.filename:
        return render_template('index.html', error="No file uploaded")

    temp_dir = tempfile.mkdtemp()

    try:
        temp_image_path = os.path.join(temp_dir, imagefile.filename)
        imagefile.save(temp_image_path)
        app.logger.info(f"File saved temporarily at {temp_image_path}")

        app.logger.info("Processing image...")
        df_test_tokens, X_test = prepro(temp_image_path)
        app.logger.info("Running prediction model...")
        hasil_bilstm_model = model(df_test_tokens, X_test)

        output_filename = f"hasil_{
            os.path.splitext(imagefile.filename)[0]}.csv"
        output_path = os.path.join("predicted_results", output_filename)
        hasil_bilstm_model.to_csv(output_path, index=False)
        app.logger.info(f"Results saved to {output_path}")

        items = hasil_bilstm_model.to_dict('records')

        return render_template('index.html', prediction=True, items=items,
                               receipt_name=imagefile.filename)

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return render_template('index.html',
                               error=f"Error processing image: {str(e)}")

    finally:
        app.logger.info(f"Cleaning up temporary directory {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

        if os.path.exists('processed_test'):
            shutil.rmtree('processed_test', ignore_errors=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
