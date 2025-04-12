import pickle
import json
from pathlib import Path

import re
import base64

import numpy as np

import tensorflow as tf

import keras
from keras import ops

class CRNN_OCR_Predictor:
    """
        CRNN (Convolutional Recurrent Neural Network) модель для распознавания текста на капчах.

        Архитектура модели:
            1. CNN-блок (извлечение визуальных признаков)
            2. RNN-блок (анализ последовательностей)
            3. CTC-слой (Connectionist Temporal Classification) для декодирования

        Основные возможности:
            - Распознавание текста на изображениях с шумами
            - Поддержка переменной длины текста
            - Работа с одноканальными (grayscale) изображениями

        Attributes:
            model (tf.keras.Model): Загруженная CRNN модель
            char_to_num (dict): Словарь маппинга символов → индексы
            num_to_char (dict): Словарь маппинга индексы → символы
            img_width (int): Ширина входного изображения
            img_height (int): Высота входного изображения
            max_length (int): Максимальная длина распознаваемого текста

        Example:
            >>> predictor = CRNN_OCR_Predictor("model_weights/")
            >>> text = predictor.predict_captcha(img)
            'A7B9K2'
    """
    
    def __init__(self, model_dir: str):
        """
            Инициализирует распознаватель капчи с загрузкой CRNN модели и вспомогательных компонентов.
            Args:
                model_dir: Путь к директории с файлами модели. Должен содержать:
                  - crnn_ocr_prediction.h5 - файл модели Keras
                  - char_mappings.pkl - словари символов
                  - config.json - параметры модели
        """
        self.model_dir = Path(model_dir)
        self._load_components()
    
    def _load_components(self):
        """
            Загружает и инициализирует все компоненты модели для распознавания капчи.

            Алгоритм загрузки:
                1. Основной CRNN-модели в формате HDF5 (.h5)
                2. Оптимизатора делегатов (XNNPACK для TFLite)
                3. Словарей маппинга символов:

                    - char_to_num: символ → числовой индекс
                    - num_to_char: числовой индекс → символ
                4. Конфигурационных параметров модели:

                    - img_width: ширина входного изображения
                    - img_height: высота входного изображения
                    - max_length: максимальная длина распознаваемого текста
        """
        # Загрузка модели
        self.model = tf.keras.models.load_model(str(self.model_dir / 'crnn_ocr_prediction.h5'))

        if hasattr(self.model, '_interpreter'):
            self.model._interpreter._delegates = [tf.lite.experimental.load_delegate('XNNPACK')]
        
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
    
    def preprocess_image_from_path(self, image_path) -> tf.Tensor:
        """
            Преобразует изображение капчи из файла в предобработанный тензор для модели CRNN.
            
            Алгоритм:
                1. Чтение файла изображения
                2. Декодирование PNG (с автоматическим приведением к grayscale)
                3. Нормализация значений пикселей в диапазон [0, 1]
                4. Проверка и коррекция размеров 
                5. Транспонирование для CTC-алгоритма
                6. Добавление batch-размерности

            Args:
                image_path: Путь к изображению (строка или Path-like объект).
                        Поддерживаемые форматы: PNG (рекомендуется), JPEG.

            Returns:
                tf.Tensor: Тензор формы (1, width, height, 1), где:
                        - 1: batch-размерность
                        - width: self.img_width (после транспонирования)
                        - height: self.img_height
                        - 1: канал (grayscale)
        """
        img = tf.io.read_file(str(image_path))
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

    def _prepare_base64(self, base64_str: str) -> str:
        """        
            Обрабатывает и валидирует строку base64 перед декодированием.
            Алгоритм:
                1. Удаляем префикс, если есть
                2. Проверяет корректность base64-данных (длина, алфавит)
            Args:
                base64_str: Исходная строка с данными в формате base64
            Returns:
                Очищенная и валидированная base64-строка, готовая к декодированию
        """
        # Удаляем префикс (если есть)
        if "," in base64_str:
            base64_str = base64_str.split(",")[-1]
        
        # Удаляем все не-base64 символы
        cleaned = re.sub(r"[^a-zA-Z0-9+/]", "", base64_str)
        
        padding = len(cleaned) % 4
        # Добавляем padding при необходимости
        if padding == 1:
            cleaned = cleaned[:-1]
        elif padding > 1:
            cleaned += "=" * (4 - padding)
         
        return cleaned
    

    def preprocess_image_from_base64(self, base64_str: str) -> tf.Tensor:
        """
            Преобразует изображение из base64 в предобработанный тензор для нейросети.

            Алгоритм:
                1. Проверка str на правильность в base64
                2. Конвертация цветового пространства в ЧБ 
                3. Изменение размера под требования модели
                4. Нормализация пикселей 
                5. Транспонирование из-за использованного CTC слоя
                6. Добавление batch-размерности 

            Args:
                base64_str: Строка в формате base64, с или без data-URI префикса
                        (data:image/<format>;base64,)

            Returns:
                tf.Tensor: Тензор с формой (batch_size, height, width, channels) 
        """

        base64_str = self._prepare_base64(base64_str)

        # Декодируем str -> base64 -> tf.Tensor
        bytes_data = base64.b64decode(base64_str)
        img = tf.io.decode_image(bytes_data, channels=1, expand_animations=False)
        
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
        return img, base64_str
     
    def predict_captcha(self, img: tf.TensorSpec ) -> str:
        """
            Распознает текст капчи из предобработанного тензора изображения.
            
            Args:
                img: Входной тензор изображения, нормализованный в [0,1]. 
                    Ожидаемая форма:
                    - Для единичного изображения: [1, height, width, channels]
                    - Для батча: [batch_size, height, width, channels]
                    Поддерживаемые размеры: высота 50px, ширина 250px, 1 канал цвета

            Returns:
                str: Распознанный текст капчи (только буквы A-Z и цифры 0-9)
                    Пустая строка означает ошибку распознавания
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

    captcha_64 = "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"

    try:
        predictor = CRNN_OCR_Predictor(MODEL_DIR)
        captch = Path('test_imgs/2mwbpf.png')
        
        print('===========\n')
        
        img = predictor.preprocess_image_from_path(captch)
        result = predictor.predict_captcha(img)
        print(result)
        
        print('===========\n')
        
        img = predictor.preprocess_image_from_base64(captcha_64)
        result = predictor.predict_captcha(img)
        print(result)

    except Exception as e:
        print(f"Ошибка: {str(e)}")



        