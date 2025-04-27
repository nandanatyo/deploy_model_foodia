from flask import Flask, render_template, request
import os
import logging
import tempfile
import shutil

# Import preprocessing dan model
from img_to_txt_ocr import prepro
from model import model

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Direktori untuk hasil prediksi saja
os.makedirs("predicted_results", exist_ok=True)

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return render_template('index.html', error="Tidak ada file yang diunggah")

    imagefile = request.files['imagefile']
    if not imagefile.filename:
        return render_template('index.html', error="Tidak ada file yang diunggah")

    # Buat direktori sementara untuk menyimpan file upload sebentar
    temp_dir = tempfile.mkdtemp()

    try:
        # Simpan file sementara
        temp_image_path = os.path.join(temp_dir, imagefile.filename)
        imagefile.save(temp_image_path)
        app.logger.info(f"File sementara disimpan di {temp_image_path}")

        # Proses gambar dan buat prediksi
        app.logger.info("Memproses gambar...")
        df_test_tokens, X_test = prepro(temp_image_path)
        app.logger.info("Menjalankan model prediksi...")
        hasil_bilstm_model = model(df_test_tokens, X_test)

        # Simpan hasil ke file CSV
        output_filename = f"hasil_{os.path.splitext(imagefile.filename)[0]}.csv"
        output_path = os.path.join("predicted_results", output_filename)
        hasil_bilstm_model.to_csv(output_path, index=False)
        app.logger.info(f"Hasil tersimpan di {output_path}")

        # Ambil hasil untuk ditampilkan
        items = hasil_bilstm_model.to_dict('records')

        return render_template('index.html', prediction=True, items=items, receipt_name=imagefile.filename)

    except Exception as e:
        app.logger.error(f"Error saat memproses gambar: {str(e)}")
        return render_template('index.html', error=f"Error saat memproses gambar: {str(e)}")

    finally:
        # Bersihkan - hapus direktori sementara dan semua isinya
        app.logger.info(f"Membersihkan direktori sementara {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Hapus juga folder processed_test jika masih ada
        if os.path.exists('processed_test'):
            shutil.rmtree('processed_test', ignore_errors=True)

if __name__ == '__main__':
    # Gunakan PORT dari environment variable yang disediakan oleh Cloud Run
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)