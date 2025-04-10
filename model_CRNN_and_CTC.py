import pickle
import json
from pathlib import Path

import numpy as np

import tensorflow as tf

import keras
from keras import ops

class CRNN_OCR_Predictor:
    def __init__(self, model_dir: str):
        """
            Класс для предсказания капчи
                :param model_dir: Путь к папке с моделью
        """
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
            self.max_length = config['max_length']
    
    def preprocess_image_for_path(self, image_path) -> tf.Tensor:
        """
            Изменение 
                :param image_path: Путь к изображению (строка или Path объект)
                :return: Обработанное изображение в нужном формате
        """
        img = tf.io.read_file(image_path)
        # Делаем чёрно-белое
        img = tf.io.decode_png(img, channels=1)
        # Приводим в нормальную форму
        img = tf.image.convert_image_dtype(img, tf.float32)

        # Получаем размер изображения
        current_shape = tf.shape(img)
        current_width = current_shape[1]
        
        # Проверяем размер изображения
        img = tf.cond(
            tf.equal(current_width, 300), # Если размер 300
            lambda: tf.image.crop_to_bounding_box(
                img,
                offset_height=0,          
                offset_width=25,          
                target_height=self.img_height,
                target_width=self.img_width    
            ),
            lambda: img 
        )
        
        img = ops.image.resize(img, [self.img_height, self.img_width])
        # Транспонирование (ВСЁ ИЗ-ЗА СTC)
        img = ops.transpose(img, axes=[1, 0, 2])
        # Мы обучались на батчах поэхтому, мы маскируем батч
        img = tf.expand_dims(img, axis=0)
        return img
     
    def predict_captcha(self, img: tf.TensorSpec ) -> str:
        """
            Предсказание капчи
                :param batch: Принимает изображение вида [None, 250, 50, 1]
                :return: Текст 
        """
        predict_float = self.model.predict(img)
        text = self._decode_batch_oneImg_predictions(predict_float)
        return text

    # A utility function to decode the output of the network
    def _decode_batch_oneImg_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        res = self._ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :self.max_length]
        # Iterate over the results and get back the text
        output_text = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
        return output_text

    def _ctc_decode(self, y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
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

    


# Пример использования
if __name__ == "__main__":
    
    MODEL_DIR = 'models/0.5v'
    TEST_IMGS_DIR = 'test_imgs'

    try:
        predictor = CRNN_OCR_Predictor(MODEL_DIR)
        img = predictor.preprocess_image_for_path('test_imgs\\2mwbpf.png')
        result = predictor.predict_captcha(img)

        print(result)
    except Exception as e:
        print(f"Ошибка: {str(e)}")