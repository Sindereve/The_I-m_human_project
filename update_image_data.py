import os

def clean_long_png_names(folder_path):
    """
        folder_path - Путь к папке для очистки.
    """
    # Проверяем, существует ли папка
    if not os.path.exists(folder_path):
        print(f"Папка {folder_path} не существует!")
        return
    
    # Получаем список всех файлов в папке
    for filename in os.listdir(folder_path):
        # Проверяем, что файл имеет расширение .png (или .PNG)
        if filename.lower().endswith('.png'):
            # Получаем имя файла без расширения
            name_without_ext = os.path.splitext(filename)[0]
            
            # Проверяем длину имени
            if len(name_without_ext) > 6:
                file_path = os.path.join(folder_path, filename)
                try:
                    os.remove(file_path)
                    print(f"Удален файл: {filename}")
                except Exception as e:
                    print(f"Ошибка при удалении {filename}: {e}")

# Пример использования
folder_to_clean = "captcha_images_v2/"  # Замените на реальный путь
clean_long_png_names(folder_to_clean)