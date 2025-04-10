import os
import pickle
import json
from pathlib import Path

import numpy as np
from PIL import Image

import tensorflow as tf

import keras
from keras import ops

class CRNN_OCR_Predictor:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self._load_components()
    
    def _load_components(self):
        """
            Загружает все необходимые компоненты модели
        """
        
        # Загрузка модели
        self.model = tf.keras.models.load_model(str(self.model_dir / 'crnn_ocr_prediction.h5'))
        
        # Загрузка словарей 
        with open(self.model_dir / 'char_mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
            self.char_to_num = mappings['char_to_num']
            self.num_to_char = mappings['num_to_char']
        
        # Загрузка конфигурации
        with open(self.model_dir / 'config.json', 'r') as f:
            config = json.load(f)
            self.img_width = config['img_width']
            self.img_height = config['img_height']
    
    def preprocess_image(self, image_path):
        """
            Подготавливает изображение для модели
                :param image_path: Путь к изображению (строка или Path объект)
                :return: Обработанное изображение в нужном формате
        """
        image_path = str(image_path)

        def encode_single_sample(img_path, label):
            # Read
            img = tf.io.read_file(img_path)
            
            # Decoding 
            img = tf.io.decode_png(img, channels=1)
            
            # Convert in normal format 0-1
            img = tf.image.convert_image_dtype(img, tf.float32)
            
            # Verifies image dimensions and resizes to target size 
            #  (img_height = 50, img_width = 250 ) 
            #  if necessary
            current_shape = tf.shape(img)
            current_width = current_shape[1]
            
            # Verifies image
            img = tf.cond(
                tf.equal(current_width, 300), # If case have 300px
                lambda: tf.image.crop_to_bounding_box(
                    img,
                    offset_height=0,          
                    offset_width=25,          # left change 
                    target_height=self.img_height,
                    target_width=self.img_width    
                ),
                lambda: img 
            )
            # Fallback: resize to default dimensions
            img = ops.image.resize(img, [self.img_height, self.img_width])
            # Transponse 
            img = ops.transpose(img, axes=[1, 0, 2])
            
            return {"image": img, "label": label}

        test_dataset = tf.data.Dataset.from_tensor_slices(([image_path], ['']))
        return (
            test_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(16)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    def decode_predictions(self, preds):
        """Декодирует предсказания модели в текст"""
        input_len = np.ones(preds.shape[0]) * preds.shape[1]
        results = tf.keras.backend.ctc_decode(
            preds, 
            input_length=input_len,
            greedy=True
        )[0][0]
        text = tf.strings.reduce_join(
            self.num_to_char(results[0])).numpy().decode('utf-8')
        return text
    
    def predict(self, image_path):
        """
        Предсказывает текст для одного изображения
            :param image_path: Путь к изображению (строка или Path объект)
            :return: Распознанный текст
        """
        try:
            processed_img = self.preprocess_image(image_path)
            preds = self.model.predict(processed_img)
            return self.decode_batch_predictions(preds)
        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {str(e)}")
            return None
        
    
    def ctc_decode(self, y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
        input_shape = ops.shape(y_pred)
        num_samples, num_steps = input_shape[0], input_shape[1]
        y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())
        input_length = ops.cast(input_length, dtype="int32")

        if greedy:
            (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
                inputs=y_pred, sequence_length=input_length
            )
        else:
            (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
                inputs=y_pred,
                sequence_length=input_length,
                beam_width=beam_width,
                top_paths=top_paths,
            )
        decoded_dense = []
        for st in decoded:
            st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
            decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
        return (decoded_dense, log_prob)
    
    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = self.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :6]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    
    def predict_folder(self, folder_path):
        """
            Обрабатывает все изображения в папке
                :param folder_path: Путь к папке (строка или Path объект)
                :return: Словарь {имя файла: распознанный текст}
        """
        folder_path = Path(folder_path)
        results = {}
        valid_extensions = ['.png']
        
        for img_file in folder_path.iterdir():
            if img_file.suffix.lower() in valid_extensions:
                try:
                    text = self.predict(img_file)
                    results[img_file.name] = text
                    print(f"Обработано: {img_file.name} -> {text}")
                except Exception as e:
                    print(f"Ошибка при обработке {img_file.name}: {str(e)}")
                    results[img_file.name] = None
        
        return results


# Пример использования
if __name__ == "__main__":
    MODEL_DIR = 'models/0.5v'
    TEST_IMGS_DIR = 'test_imgs'
    
    try:
        predictor = CRNN_OCR_Predictor(MODEL_DIR)
        results = predictor.predict_folder(TEST_IMGS_DIR)
        
        print("\nРезультаты распознавания:")
        for filename, text in results.items():
            print(f"{filename}: {text[0]}")
            
    except Exception as e:
        print(f"Ошибка: {str(e)}")