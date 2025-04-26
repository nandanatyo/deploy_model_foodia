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
    from nltk.corpus import stopwords
    from paddleocr import PaddleOCR,draw_ocr
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from collections import defaultdict

    def process_receipt(image_path, show=True):
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

        # Optional Visualization
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


    def preprocess_receipts(input_dir, output_dir, process_receipt_fn, show=False):
        """
        Memproses semua gambar struk dalam folder input_dir dan menyimpannya di output_dir.

        Args:
            input_dir (str): Path ke folder input (berisi gambar struk).
            output_dir (str): Path ke folder output hasil preprocessing.
            process_receipt_fn (function): Fungsi untuk memproses struk, return (cropped, enhanced).
            show (bool): Jika True, tampilkan hasil tiap proses (opsional).

        Returns:
            list: Path dari file hasil enhanced.
        """
        os.makedirs(output_dir, exist_ok=True)

        valid_exts = (".jpg", ".jpeg", ".png")
        receipt_paths = [
            # os.path.join(input_dir, fname)
            # for fname in os.listdir(input_dir)
            # if fname.lower().endswith(valid_exts)
            input_dir
        ]

        processed_files = []
        for path in receipt_paths:
            cropped, enhanced = process_receipt_fn(path, show=show)
            base_name = os.path.splitext(os.path.basename(path))[0]
            cropped_path = os.path.join(output_dir, f"{base_name}_cropped.jpg")
            enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.jpg")
            cv2.imwrite(enhanced_path, enhanced)
            processed_files.append(enhanced_path)

        # print(f"Jumlah struk diproses dari {input_dir}: {len(processed_files)}")
        return processed_files

    # test_dir = '/images'
    test_out = '/processed_test'
    test_files = preprocess_receipts(image, test_out, process_receipt_fn=process_receipt)

    ocr = PaddleOCR(use_angle_cls=True, lang='id') # need to run only once to download and load model into memory

    def run_ocr_on_receipts(processed_paths, ocr_model, use_cls=True):
        """
        Jalankan OCR pada daftar path gambar yang telah diproses.

        Args:
            processed_paths (list): Daftar path ke gambar hasil preprocessing.
            ocr_model (PaddleOCR): Objek PaddleOCR yang sudah diinisialisasi di luar fungsi.
            use_cls (bool): Gunakan klasifikasi orientasi teks.

        Returns:
            list of dict: Setiap dict berisi 'path' dan 'ocr_result'.
        """
        ocr_results = []
        for path in processed_paths:
            result = ocr_model.ocr(path, cls=True)
            ocr_results.append({
                'path': path,
                'ocr_result': result
            })

        # print(f"OCR selesai untuk {len(ocr_results)} gambar.")
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

            # Print hasil (optional)
            # print(f"\n=== Struk {idx+1} ({entry['path']}) ===")
            # for i, line in enumerate(grouped_lines):
            #     line_text = ' '.join([item[1] for item in line])
            #     print(f"Baris {i+1}: {line_text}")

        return all_grouped

    semua_struk_dalam_baris_test = process_all_receipts(test_ocr_results)

    # --- Unduh stopwords Bahasa Indonesia jika belum tersedia ---
    nltk.download('stopwords')
    stop_words = set(stopwords.words('indonesian'))

    # --- Fungsi Preprocessing Token ---
    def preprocess_receipt_tokens(semua_struk_dalam_baris, apply_stopwords=True):
        """
        Ekstrak dan bersihkan token dari grouped_lines hasil OCR.

        Args:
            semua_struk_dalam_baris (list): List of dict hasil OCR, berisi path dan grouped_lines.
            apply_stopwords (bool): True jika ingin menghapus stopwords Bahasa Indonesia.

        Returns:
            pd.DataFrame: Kolom filename, sentence_id, dan token.
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

        Args:
            df (DataFrame): Data test tanpa label (kolom: filename, sentence_id, token)
            tokenizer (Tokenizer): Tokenizer hasil training (sudah di-load)
            max_len (int): Panjang padding yang konsisten dengan training

        Returns:
            X_test (np.array): Data test siap input ke model
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

    return df_test_tokens, X_test