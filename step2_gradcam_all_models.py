import tensorflow as tf

models_folder = r"C:\fyp\h5.files"

model_names = ['ResNet50', 'VGG16', 'DenseNet121', 'MobileNetV2', 'InceptionV3']

print("Checking models...\n")
for name in model_names:
    try:
        path = f"{models_folder}\\{name}_final.keras"
        # ✅ Fix — add safe_mode=False
        model = tf.keras.models.load_model(path, safe_mode=False)
        print(f"✅ {name} — OK")
        print(f"   Input shape:  {model.input_shape}")
        print(f"   Output shape: {model.output_shape}\n")
    except Exception as e:
        print(f"❌ {name} — CORRUPTED or NOT FOUND")
        print(f"   Error: {e}\n")