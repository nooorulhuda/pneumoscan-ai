import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# =============================================================
# PATHS
# =============================================================
MODELS_FOLDER = r'C:\fyp\h5.files'
TEST_DIR      = r'C:\fyp\chest_xray\test'

# =============================================================
# BUILD MODEL
# =============================================================
def build_model(builder, img_size):
    inputs     = keras.Input(shape=(img_size, img_size, 3))
    base_model = builder(input_shape=(img_size, img_size, 3),
                         include_top=False, weights=None)
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs=inputs, outputs=x)

# =============================================================
# MODEL CONFIGS
# =============================================================
MODEL_CONFIGS = {
    'ResNet50': {
        'builder':    keras.applications.ResNet50,
        'preprocess': keras.applications.resnet50.preprocess_input,
        'img_size':   224,
    },
    'VGG16': {
        'builder':    keras.applications.VGG16,
        'preprocess': keras.applications.vgg16.preprocess_input,
        'img_size':   224,
    },
    'DenseNet121': {
        'builder':    keras.applications.DenseNet121,
        'preprocess': keras.applications.densenet.preprocess_input,
        'img_size':   224,
    },
    'MobileNetV2': {
        'builder':    keras.applications.MobileNetV2,
        'preprocess': keras.applications.mobilenet_v2.preprocess_input,
        'img_size':   224,
    },
    'InceptionV3': {
        'builder':    keras.applications.InceptionV3,
        'preprocess': keras.applications.inception_v3.preprocess_input,
        'img_size':   299,
    },
}

# =============================================================
# RUN
# =============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (model_name, config) in enumerate(MODEL_CONFIGS.items()):
    print(f"\n🔄 Processing {model_name}...")
    img_size = config['img_size']

    # Load test data
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(img_size, img_size),
        batch_size=32,
        shuffle=False,
        label_mode='binary'
    )

    # Build and load model
    model        = build_model(config['builder'], img_size)
    weights_path = os.path.join(MODELS_FOLDER, f'{model_name}_best.weights.h5')
    model.load_weights(weights_path)

    # Predictions
    y_true, y_pred = [], []
    for images, labels in test_ds:
        images_proc = config['preprocess'](images.numpy().copy())
        preds       = model.predict(images_proc, verbose=0)
        y_pred.extend((preds > 0.5).astype(int).flatten())
        y_true.extend(labels.numpy().astype(int).flatten())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'],
                ax=axes[idx], cbar=False,
                annot_kws={'size': 16, 'weight': 'bold'})

    axes[idx].set_title(f'{model_name}', fontsize=14,
                        fontweight='bold', pad=10)
    axes[idx].set_xlabel('Predicted', fontsize=11)
    axes[idx].set_ylabel('Actual', fontsize=11)

    print(f"✅ {model_name} done!")
    tf.keras.backend.clear_session()

# Hide last empty subplot
axes[5].set_visible(False)

plt.suptitle('Confusion Matrices — All 5 Models',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(r'C:\fyp\results\confusion_matrices_all_models.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Saved to C:\\fyp\\results\\confusion_matrices_all_models.png")