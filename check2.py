import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

print("TF version:", tf.__version__)

models_folder = r"C:\fyp\h5.files"

MODEL_CONFIGS = {
    'ResNet50':    (keras.applications.ResNet50,    224),
    'VGG16':       (keras.applications.VGG16,       224),
    'DenseNet121': (keras.applications.DenseNet121, 224),
    'MobileNetV2': (keras.applications.MobileNetV2, 224),
    'InceptionV3': (keras.applications.InceptionV3, 299),
}

def build_model(builder, img_size):
    base_model = builder(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=None  # no imagenet weights needed, just architecture
    )
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

print("\nChecking models...\n")
for name, (builder, img_size) in MODEL_CONFIGS.items():
    try:
        # ✅ Rebuild architecture first
        model = build_model(builder, img_size)

        # ✅ Then load just the weights
        weights_path = f"{models_folder}\\{name}_best.weights.h5"
        model.load_weights(weights_path)

        print(f"✅ {name} — OK!")
        print(f"   Input:  {model.input_shape}")
        print(f"   Output: {model.output_shape}\n")

    except Exception as e:
        print(f"❌ {name} — Error: {e}\n")
        