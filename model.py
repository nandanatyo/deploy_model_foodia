def model(df_test_tokens, X_test):
    import pandas as pd
    import pickle
    import numpy as np
    import re
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from collections import defaultdict

    model_save = load_model('bilstm_model2.h5')

    with open("save_label_encoder2.pkl", "rb") as f:
        save_label_encoder = pickle.load(f)

    # Prediksi probabilitas
    y_test_proba = model_save.predict(X_test)

    # Ambil index label prediksi
    y_test_idx = np.argmax(y_test_proba, axis=-1)

    # Ubah ke label asli
    y_test_labels = save_label_encoder.inverse_transform(
        y_test_idx.ravel()).reshape(y_test_idx.shape)

    def predict_test_labels(test_df, y_test_labels, max_len):
        # Kelompokkan token berdasarkan (filename, sentence_id)
        tokens_grouped = defaultdict(list)
        for _, row in test_df.iterrows():
            key = (row['filename'], row['sentence_id'])
            tokens_grouped[key].append(row['token'])

        # Susun hasil prediksi
        predicted_data = []
        for i, key in enumerate(tokens_grouped):
            tokens = tokens_grouped[key]
            preds = y_test_labels[i][:len(tokens)]  # potong padding
            for j, token in enumerate(tokens):
                predicted_data.append({
                    'filename': key[0],
                    'sentence_id': key[1],
                    'token': token,
                    'predicted_label': preds[j]
                })

        return pd.DataFrame(predicted_data)

    def is_valid_price(text):
        # True jika mengandung angka atau angka dengan titik/koma
        return bool(re.fullmatch(r"[\d.,]+", text))

    def extract_items_and_prices_flexible(df):
        extracted = []

        grouped = df.groupby(["filename", "sentence_id"])

        for (filename, sentence_id), group in grouped:
            tokens = group["token"].tolist()
            labels = group["predicted_label"].tolist()

            items = []
            prices = []
            i = 0

            while i < len(labels):
                # Deteksi item
                if labels[i] in ["B-ITEM_NAME", "I-ITEM_NAME"]:
                    item = tokens[i]
                    i += 1
                    while i < len(labels) and labels[i] == "I-ITEM_NAME":
                        item += " " + tokens[i]
                        i += 1
                    items.append(item)

                # Deteksi price
                elif labels[i] in ["B-PRICE", "I-PRICE"]:
                    price = tokens[i]
                    i += 1
                    while i < len(labels) and labels[i] == "I-PRICE":
                        price += " " + tokens[i]
                        i += 1
                    # Simpan hanya jika valid price
                    if is_valid_price(price.replace(" ", "")):
                        prices.append(price)
                else:
                    i += 1

            # Pasangkan
            if len(items) == 1 and len(prices) == 1:
                extracted.append({
                    "filename": filename,
                    "sentence_id": sentence_id,
                    "ITEM_NAME": items[0],
                    "PRICE": prices[0]
                })
            elif len(items) > 0:
                for item in items:
                    price = prices.pop(0) if len(prices) > 0 else np.nan
                    extracted.append({
                        "filename": filename,
                        "sentence_id": sentence_id,
                        "ITEM_NAME": item,
                        "PRICE": price
                    })

        return pd.DataFrame(extracted)

    df_test_predictions = predict_test_labels(
        test_df=df_test_tokens,
        y_test_labels=y_test_labels,
        max_len=16
    )

    return extract_items_and_prices_flexible(df_test_predictions)
