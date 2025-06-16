import os
import shutil
import random

# Configuración
input_dir = "C:\\Users\\Augusto\\Documents\\entidad-contraataque\\mision3\\src\\main\\resources\\dataset"
output_dir = "C:\\Users\\Augusto\\Documents\\entidad-contraataque\\mision3\\src\\main\\resources\\dataset_final"              # Carpeta donde quedarán train/test
split_ratio = 0.7                   # 80% train, 20% test
random_seed = 123                   # Para reproducibilidad

# Clases (subcarpetas en input_dir)
classes = os.listdir(input_dir)

for subset in ['train', 'test']:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, subset, cls), exist_ok=True)

# Procesar cada clase por separado
for cls in classes:
    class_dir = os.path.join(input_dir, cls)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    random.Random(random_seed).shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]

    for img in train_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(output_dir, 'train', cls, img)
        shutil.copy2(src, dst)

    for img in test_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(output_dir, 'test', cls, img)
        shutil.copy2(src, dst)

    print(f"[{cls}] → Total: {len(images)} | Train: {len(train_images)} | Test: {len(test_images)}")

print("✅ División 80/20 completada.")
