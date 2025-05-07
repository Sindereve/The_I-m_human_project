# Распознавание текста с помощью CRNN и CTC + автоматизация тестирования сайта

Проект сочетает обучение модели распознавания текста с использованием архитектуры CRNN и функции потерь CTC, а также автоматизацию тестирования модели на сайте с капчей с помощью Selenium.

## 📁 Структура проекта

```text
root/
├── models/                     # Папка с обученными моделями
│   ├── 0.5v/                   # Модель версии 0.5 модель для предсказания 6 знаков в капче
│   └── 0.8v/                   # Модель версии 0.8 модель для предсказания 10 знаков в капче
├── download_data.ipynb         # Блокнот для загрузки данных
├── model_create.ipynb          # Блокнот для обучения модели (CRNN + CTC Loss)
├── model_CRNN_and_CTC.py       # Модуль с определением модели
├── testing.py                  # Скрипт с Selenium для обхода капчи
├── requirements.txt            # txt с нужными библотеками
```

## 🧠 Описание

- **Модель**: Используется CRNN (Convolutional Recurrent Neural Network) с функцией потерь CTC (Connectionist Temporal Classification) для распознавания текста на изображениях.
- **Selenium-тестирование**: `testing.py` автоматически открывает сайт, обновляет его и взаимодействует с элементами капчи.

## 🖼️ Примеры работы

### 🔍 Распознавание текста моделью:

![Пример работы модели](https://github.com/user-attachments/assets/fb537ad4-14d4-4bc9-8faa-e393cce2ada7)
![Пример работы модели](https://github.com/user-attachments/assets/2be4cd06-355e-42bc-be43-593fccc36f5f)

## ⚙️ Как запустить

1. **Склонируйте репозиторий:**
```bash
git clone https://github.com/Sindereve/The_I-m_human_project.git
https://github.com//edit/main/README.md
cd your_repo
```
2. **Склонируйте репозиторий:**
```bash
pip install -r requirements.txt
```
3. **Загрузите данные:**
Откройте и выполните download_data.ipynb
4. **Обучите модель:**
Выполните блокнот model_create.ipynb для создания и обучения модели.
5. **Запустите тестирование сайта:**
```bash
python testing.py
```

## Примечания
* Для работы testing.py требуется установленный WebDriver (например, ChromeDriver) и соответствующий браузер.
* Возможно, сайт меняет структуру, поэтому локаторы в Selenium могут потребовать адаптации.
