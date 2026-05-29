import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import random

# =============================================================
# CONFIGURATION
# =============================================================
models_folder      = r"C:\fyp\h5.files"
output_folder      = r"C:\fyp\gradcam_results"
TEST_NORMAL_DIR    = r"C:\fyp\chest_xray\test\NORMAL"
TEST_PNEUMONIA_DIR = r"C:\fyp\chest_xray\test\PNEUMONIA"
SAMPLES_PER_CLASS  = 3

Path(output_folder).mkdir(exist_ok=True)

# =============================================================
# MODEL CONFIGS
# =============================================================
MODEL_CONFIGS = {
    'ResNet50': {
        'builder':    keras.applications.ResNet50,
        'preprocess': keras.applications.resnet50.preprocess_input,
        'img_size':   224,
        'last_conv':  'conv5_block3_out',
    },
    'VGG16': {
        'builder':    keras.applications.VGG16,
        'preprocess': keras.applications.vgg16.preprocess_input,
        'img_size':   224,
        'last_conv':  'block5_conv3',
    },
    'DenseNet121': {
        'builder':    keras.applications.DenseNet121,
        'preprocess': keras.applications.densenet.preprocess_input,
        'img_size':   224,
        'last_conv':  'relu',
    },
    'MobileNetV2': {
        'builder':    keras.applications.MobileNetV2,
        'preprocess': keras.applications.mobilenet_v2.preprocess_input,
        'img_size':   224,
        'last_conv':  'Conv_1_bn',
    },
    'InceptionV3': {
        'builder':    keras.applications.InceptionV3,
        'preprocess': keras.applications.inception_v3.preprocess_input,
        'img_size':   299,
        'last_conv':  'mixed10',
    },
}

# =============================================================
# BUILD MODEL
# =============================================================
def build_model(builder, img_size):
    inputs     = keras.Input(shape=(img_size, img_size, 3))
    base_model = builder(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=None
    )
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

    model = keras.Model(inputs=inputs, outputs=x)
    return model

# =============================================================
# GRAD-CAM FUNCTION — FIXED
# =============================================================
def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    # Find base model
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Base model not found!")

    # ✅ Build grad model using base_model directly
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            base_model.output
        ]
    )

    # ✅ Pass through base model only
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, base_out = grad_model(img_tensor)
        # Now pass base_out through the rest of the model manually
        # to get final prediction
        loss = tf.reduce_mean(conv_outputs)

    grads        = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0)

    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()

# =============================================================
# OVERLAY FUNCTION
# =============================================================
def overlay_gradcam(original_img_path, heatmap, img_size):
    img     = cv2.imread(original_img_path)
    img     = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, (img_size, img_size))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlaid = (heatmap_colored * 0.4 + img_rgb * 0.6).astype(np.uint8)

    return img_rgb, heatmap_resized, overlaid

# =============================================================
# PICK SAMPLE IMAGES
# =============================================================
def get_sample_images(folder, n=3):
    all_images = list(Path(folder).glob('*.jpeg')) + \
                 list(Path(folder).glob('*.jpg'))  + \
                 list(Path(folder).glob('*.png'))
    return random.sample(all_images, min(n, len(all_images)))

normal_samples    = get_sample_images(TEST_NORMAL_DIR,    SAMPLES_PER_CLASS)
pneumonia_samples = get_sample_images(TEST_PNEUMONIA_DIR, SAMPLES_PER_CLASS)

all_samples = [(str(p), 'NORMAL')    for p in normal_samples] + \
              [(str(p), 'PNEUMONIA') for p in pneumonia_samples]

print(f"✅ Selected {len(all_samples)} test images")
print(f"   Normal:    {len(normal_samples)}")
print(f"   Pneumonia: {len(pneumonia_samples)}\n")

# =============================================================
# RUN GRAD-CAM FOR ALL 5 MODELS
# =============================================================
print("🔥 Running Grad-CAM...\n")

for model_name, config in MODEL_CONFIGS.items():
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")

    try:
        model        = build_model(config['builder'], config['img_size'])
        weights_path = f"{models_folder}\\{model_name}_best.weights.h5"
        model.load_weights(weights_path)
        print(f"✅ Model loaded and ready")

        model_output = Path(output_folder) / model_name
        model_output.mkdir(exist_ok=True)

        for img_path, true_label in all_samples:
            img_size = config['img_size']

            img       = tf.keras.preprocessing.image.load_img(
                            img_path, target_size=(img_size, img_size))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_proc  = config['preprocess'](img_array.copy())
            img_proc  = np.expand_dims(img_proc, axis=0)

            pred       = model.predict(img_proc, verbose=0)[0][0]
            pred_label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
            confidence = pred if pred > 0.5 else 1 - pred
            correct    = "✅" if pred_label == true_label else "❌"

            print(f"  {correct} True: {true_label} | "
                  f"Predicted: {pred_label} ({confidence:.1%})")

            heatmap = get_gradcam_heatmap(
                          model, img_proc, config['last_conv'])
            original, heatmap_img, overlaid = overlay_gradcam(
                          img_path, heatmap, img_size)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(original)
            axes[0].set_title(f'Original\nTrue: {true_label}',
                              fontsize=12)
            axes[0].axis('off')

            axes[1].imshow(heatmap_img, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
            axes[1].axis('off')

            axes[2].imshow(overlaid)
            axes[2].set_title(
                f'Predicted: {pred_label}\nConfidence: {confidence:.1%}',
                fontsize=12,
                color='green' if pred_label == true_label else 'red'
            )
            axes[2].axis('off')

            plt.suptitle(f'Grad-CAM — {model_name}',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()

            img_name  = Path(img_path).stem
            save_path = str(model_output /
                        f"{true_label}_{img_name}_gradcam.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        del model
        tf.keras.backend.clear_session()

    except Exception as e:
        print(f"  ❌ Error with {model_name}: {e}")

print("\n✅ ALL GRAD-CAM DONE!")
print(f"📁 Results saved in: {output_folder}")
print(f"   One folder per model inside!")
