def prepro(image):
    import cv2
    import imutils
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import shutil
    import re
    import pandas as pd
    import nltk
    import pickle
    import tempfile
    from nltk.corpus import stopwords
    from paddleocr import PaddleOCR,draw_ocr
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from collections import defaultdict

    def process_receipt(image_path, show=False):
        def orient_vertical(img):
            h, w = img.shape[:2]
            return imutils.rotate(img, angle=270) if w > h else img

        def sharpen_edge(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
            dilated = cv2.dilate(blurred, rectKernel, iterations=2)
            edged = cv2.Canny(dilated, 75, 200, apertureSize=3)
            return edged

        def binarize(gray):
            return cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, 25, 15
            )

        def find_receipt_bounding_box(binary, img):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return img, None, None
            largest_cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_cnt)
            box = np.intp(cv2.boxPoints(rect))
            boxed = cv2.drawContours(img.copy(), [box], 0, (0, 255, 0), 5)
            return boxed, largest_cnt, rect

        def adjust_tilt(img, angle):
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            rotated = imutils.rotate_bound(img, -angle)
            return rotated, -angle

        def crop(img, contour):
            x, y, w, h = cv2.boundingRect(contour)
            return img[y:y+h, x:x+w]

        def enhance_txt_adaptive(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 25, 15
            )

        # 1. Load & rotate if landscape
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            raise ValueError(f"Failed to load image at {image_path}")
        rotated = orient_vertical(raw_img)

        # 2. Binarize for bounding box
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        binary = binarize(gray)

        # 3. Bounding box
        boxed, largest_cnt, rect = find_receipt_bounding_box(binary, rotated.copy())
        if largest_cnt is None:
            print("No receipt detected.")
            return None, None

        # 4. Tilt correction
        angle = rect[-1]
        tilted, corrected_angle = adjust_tilt(rotated.copy(), angle)

        # 5. Cari ulang contour di image yang sudah diluruskan
        gray_tilted = cv2.cvtColor(tilted, cv2.COLOR_BGR2GRAY)
        _, binary_tilted = cv2.threshold(gray_tilted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_tilted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_cnt_tilted = max(contours, key=cv2.contourArea)

        # 6. Crop
        cropped = crop(tilted, largest_cnt_tilted)

        # 7. Enhance for OCR
        enhanced = enhance_txt_adaptive(cropped)

        # Optional Visualization (disabled in this version)
        if show:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
            plt.title("Original / Rotated")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            plt.title("Cropped Receipt")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(enhanced, cmap='gray')
            plt.title("Enhanced for OCR")
            plt.axis('off')
            plt.show()

        return cropped, enhanced


    def preprocess_receipts(input_dir, process_receipt_fn, show=False):
        """
        Memproses gambar struk dan menyimpan hasil sementara.
        """
        # Buat direktori sementara
        temp_dir = tempfile.mkdtemp(prefix="processed_")

        cropped, enhanced = process_receipt_fn(input_dir, show=show)

        if cropped is None or enhanced is None:
            return []

        # Simpan enhanced ke file sementara
        base_name = os.path.splitext(os.path.basename(input_dir))[0]
        enhanced_path = os.path.join(temp_dir, f"{base_name}_enhanced.jpg")
        cv2.imwrite(enhanced_path, enhanced)

        return [enhanced_path]

    # Buat direktori sementara untuk output OCR
    test_out = tempfile.mkdtemp(prefix="processed_")
    test_files = preprocess_receipts(image, process_receipt_fn=process_receipt)

    if not test_files:
        raise ValueError("Tidak dapat memproses gambar")

    ocr = PaddleOCR(use_angle_cls=True, lang='id') # need to run only once to download and load model into memory

    def run_ocr_on_receipts(processed_paths, ocr_model, use_cls=True):
        """
        Jalankan OCR pada daftar path gambar yang telah diproses.
        """
        ocr_results = []
        for path in processed_paths:
            result = ocr_model.ocr(path, cls=True)
            ocr_results.append({
                'path': path,
                'ocr_result': result
            })

        return ocr_results

    test_ocr_results = run_ocr_on_receipts(test_files, ocr)

    def group_text_by_line(single_result, y_threshold=18):
        """
        Mengelompokkan hasil OCR satu struk menjadi baris-baris teks berdasarkan posisi y-nya.
        """
        # Hasil dari PaddleOCR satu struk = list of (box, (text, score))
        sorted_items = sorted(single_result, key=lambda x: x[0][0][1])  # sort top to bottom

        lines = []
        current_line = []
        last_y = None

        for box, (text, score) in sorted_items:
            y_top = box[0][1]
            if last_y is None:
                current_line.append((box, text))
                last_y = y_top
            elif abs(y_top - last_y) <= y_threshold:
                current_line.append((box, text))
            else:
                current_line = sorted(current_line, key=lambda b: b[0][0][0])
                lines.append(current_line)
                current_line = [(box, text)]
                last_y = y_top

        if current_line:
            current_line = sorted(current_line, key=lambda b: b[0][0][0])
            lines.append(current_line)

        return lines

    def process_all_receipts(ocr_results, y_threshold=18):
        """
        Memproses hasil OCR dari banyak struk menjadi baris-baris teks yang rapi.
        """
        all_grouped = []

        for idx, entry in enumerate(ocr_results):
            result = entry['ocr_result']  # ambil result dari dict
            grouped_lines = group_text_by_line(result[0], y_threshold)  # result[0] = satu halaman

            all_grouped.append({
                'path': entry['path'],
                'grouped_lines': grouped_lines
            })

        return all_grouped

    semua_struk_dalam_baris_test = process_all_receipts(test_ocr_results)

    # --- Unduh stopwords Bahasa Indonesia jika belum tersedia ---
    nltk.download('stopwords')
    stop_words = set(stopwords.words('indonesian'))

    # --- Fungsi Preprocessing Token ---
    def preprocess_receipt_tokens(semua_struk_dalam_baris, apply_stopwords=True):
        """
        Ekstrak dan bersihkan token dari grouped_lines hasil OCR.
        """
        all_token_info = []

        for struk in semua_struk_dalam_baris:
            path = struk['path']
            grouped_lines = struk['grouped_lines']
            filename = path.split("/")[-1]

            for sentence_id, line in enumerate(grouped_lines, 1):
                full_line = ' '.join([item[1] for item in line])
                full_line = full_line.lower()
                full_line = re.sub(r'[^a-z0-9\s]', '', full_line)  # Hapus karakter non-alfanumerik
                tokens = full_line.strip().split()

                if apply_stopwords:
                    tokens = [t for t in tokens if t not in stop_words]

                for token in tokens:
                    all_token_info.append({
                        "filename": filename,
                        "sentence_id": sentence_id,
                        "token": token
                    })

        return pd.DataFrame(all_token_info)

    # df_test_tokens untuk data uji (tidak akan dilabeli BIO)
    df_test_tokens = preprocess_receipt_tokens(semua_struk_dalam_baris_test)

    # Load tokenizer
    with open('save_tokenizer2.pkl', 'rb') as f:
        save_tokenizer = pickle.load(f)

    def prepare_test_data(df, tokenizer, max_len):
        """
        Siapkan data test untuk prediksi BiLSTM.
        """
        grouped = defaultdict(list)

        for _, row in df.iterrows():
            key = (row['filename'], row['sentence_id'])
            grouped[key].append(row['token'])

        sequences = list(grouped.values())
        X_tokenized = tokenizer.texts_to_sequences(sequences)
        X_padded = pad_sequences(X_tokenized, maxlen=max_len, padding='post')

        return X_padded

    X_test = prepare_test_data(df_test_tokens, tokenizer=save_tokenizer, max_len=16)

    # Bersihkan file-file sementara
    for file_path in test_files:
        try:
            os.remove(file_path)
        except:
            pass

    try:
        # Bersihkan direktori sementara
        shutil.rmtree(test_out, ignore_errors=True)
    except:
        pass

    return df_test_tokens, X_test