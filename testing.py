from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import base64
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from typing import Literal, Optional

import time
from datetime import datetime
from model_CRNN_and_CTC import CRNN_OCR_Predictor

# задаём ожидание после предсказания капчи
time_min = 1


def save_captcha_image(base64_data: str, label: str, save_dir: str = "captcha_history") -> None:
    """
        Декодирует и сохраняет изображение капчи из base64 в указанную директорию.
        
        Функция создает директорию, если она не существует, и сохраняет изображение
        в формате 'mr3lk9.png'. Метка очищается от недопустимых символов
        для использования в имени файла.

        Args:
            base64_data: Строка с данными изображения в формате base64.
            label: Правильный текст капчи, который будет использован в имени файла.
            save_dir: Целевая директория для сохранения (по умолчанию 'captcha_history').
    """
    try:
        # Создаем директорию, если её нет
        os.makedirs(save_dir, exist_ok=True)
        
        # называем нашу модель
        filename = f"{label}.png"
        filepath = os.path.join(save_dir, filename)
        
        # Декодируем и сохраняем
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(base64_data))
        
        print(f"[{datetime.now()}] Капча сохранена: {filepath}")
    
    except Exception as e:
        print(f"[{datetime.now()}] Ошибка сохранения капчи: {e}")

def check_for_сaptcha(driver: webdriver) -> Optional[str]:
    """
        Проверяет наличие капчи на текущей веб-странице и извлекает ее данные.

        Args:
            driver: Экземпляр Selenium WebDriver для взаимодействия со страницей.

        Returns:
            Optional[str]:
            - Если капча обнаружена: возвращает текстовое содержимое/идентификатор капчи
            - Если капча отсутствует: возвращает None
    """
    try:
        captcha_div = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@style, 'base64')]"))
        )
        return captcha_div.get_attribute("style")
    except Exception as e:
        print(f"[{datetime.now()}] Капчи нету!!!")
        return False

def click_verification_predict(driver: webdriver, predict_str: str) -> None:
    """
        Вводит предсказанный текст капчи в веб-форму и отправляет её.
        Args:
            driver: Экземпляр Selenium WebDriver для взаимодействия со страницей
            predict_str: Предсказанный текст капчи для ввода
        Raises:
            NoSuchElementException: Если не найдены элементы:
                                - Поле ввода (ID: appointment_captcha_month_captchaText)
                                - Кнопка отправки (ID: appointment_captcha_month_appointment_showMonth)
            WebDriverException: При проблемах взаимодействия с элементами
    """
    try:
        # Ввод капчи
        input_field = driver.find_element(By.ID, "appointment_captcha_month_captchaText")
        input_field.clear()
        input_field.send_keys(predict_str)
        # Клик по кнопке "отправить"
        submit_button = driver.find_element(By.ID, "appointment_captcha_month_appointment_showMonth")
        submit_button.click()
        print(f"[{datetime.now()}] Форма отправлена.")
    except Exception as e:
        print(f"[{datetime.now()}] Ошибка с отправкой предсказания капчи!!!")



def solve_captcha(driver: webdriver, predictor: CRNN_OCR_Predictor) -> Literal["passed", "failed", "no_captcha"]:
    """
    Распознает и вводит капчу на странице с помощью ML-модели.

    Алгоритм:
        1. Проверяет наличие капчи на странице.
        2. Если капча есть — извлекает изображение и отправляет в модель.
        3. Вводит предсказанный текст и проверяет успешность решения.

    Args:
        driver: Экземпляр Selenium WebDriver для взаимодействия со страницей.
        predictor: Модель CRNN для распознавания текста капчи.

    Returns:
        - "passed" — капча успешно решена и пройдена.
        - "failed" — капча обнаружена, но решение неверное.
        - "no_captcha" — капча отсутствует на странице.
    """

    # Проверка наличия капчи
    data_for_сaptcha  = check_for_сaptcha(driver)
    
    if data_for_сaptcha:

        # Предсказание капчи
        img, base64_data = predictor.preprocess_image_from_base64(data_for_сaptcha)
        predict_str = predictor.predict_captcha(img).strip()
        
        time.sleep(10) # сон для мужиков !!!
        # дело в том, что система говорит, что мы не верно ввели капчу
        # хотя капча введена верно .... Поэтому мы добавим элемент человека
        print(f"[{datetime.now()}] На капче написано:{predict_str}")

        # Отправка нашего предикта
        click_verification_predict(driver, predict_str)
            
        # Проврка, верно ли мы отгадываем капчу :)
        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, "//div[@id='message' and contains(., 'Текст введен неверно')]"))
            )
            print(f"[{datetime.now()}] Ошибка: капча не пройдена!")
            save_captcha_image(base64_data, predict_str, 'test_false')
            
            return "failed"  # Не верно решили капчу
        except:
            print(f"[{datetime.now()}] Капча успешно пройдена.")
            save_captcha_image(base64_data, predict_str, 'test_true')
            
            return "passed"  # Верно решили капчу
        
    else:
        return "no_captcha" # Капчи нету
        

MODEL_DIR = 'models/0.5v'

def main():
    # Настройка драйвера
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(options=options)
    
    predictor = CRNN_OCR_Predictor(MODEL_DIR)

    try:
        # Количество верно решённых капч
        count_true = 0
        # Количество не верно решённых капч
        count_false = 0 
        # Количество не верно подряд решённых капч
        false_ultra = 0

        while True:
            print(f"\n[{datetime.now()}] Начало новой проверки...")
            driver.get("https://service2.diplo.de/rktermin/extern/appointment_showMonth.do?locationCode=niko&realmId=926&categoryId=1955&dateStr=12.04.2025")
            
            # Проверка на заход в сложные капчи
            if false_ultra >= 5:
                print(' __Сложные капчи__ ')
                print('...Ждём 5 минут...')
                false_ultra = 0
                time.sleep(300)

            # Ожидание загрузки страницы и обработка капчи
            captcha_result_predict = solve_captcha(driver, predictor)
            
            if captcha_result_predict == 'passed':
                # Капча решена верно

                # Проверка открыт ли сайт
                try:
                    # тут сейчас заглушка по сути
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Доступные даты')]"))
                    )
                    print(f"[{datetime.now()}] Найдены доступные даты!")
                except:
                    print(f"[{datetime.now()}] Нет доступных дат.")

                # Пауза перед следующим запросом
                count_true+=1
                false_ultra = 0

                print(f"[{datetime.now()}] Ожидание {time_min} минут...")
                print(f"+{count_true} -{count_false}")
                time.sleep(time_min*60)
            elif captcha_result_predict == 'failed':
                # Капча решена не верно
                count_false+=1
                false_ultra+=1
            else:
                # Капчи нету
                # Пауза перед следующим запросом
                print(f"[{datetime.now()}] Ожидание {time_min} минут...")
                time.sleep(time_min*60)
            
    except KeyboardInterrupt:
        print("\nСкрипт остановлен вручную.")
        print(f"Статистика:")
        print(f" - Верно решено: {count_true}")
        print(f" - Не верно решено: {count_false}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()