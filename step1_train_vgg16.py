import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("STEP 1: TRAINING VGG16 MODEL ON CHEST X-RAY DATASET")
print("=" * 80)

# ============================================================================
# PATHS
# ============================================================================
TRAIN_DIR = r'C:\fyp\chest_xray\train'
VAL_DIR = r'C:\fyp\chest_xray\val'
OUTPUT_DIR = r'C:\fyp\h5.files'

print(f"\n📁 Checking paths...")
print(f"   Training data: {TRAIN_DIR}")
print(f"   Validation data: {VAL_DIR}")
print(f"   Output directory: {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001

print(f"\n⚙️ Hyperparameters:")
print(f"   Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Epochs: {EPOCHS}")
print(f"   Learning rate: {LEARNING_RATE}")

# ============================================================================
# 1. DATA LOADING WITH AUGMENTATION
# ============================================================================
print("\n📊 Setting up data loaders...")

# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data generator (only rescaling)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load training data
print("   Loading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',  # 0=Normal, 1=Pneumonia
    color_mode='rgb'
)

# Load validation data
print("   Loading validation data...")
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb'
)

print(f"✅ Data loaders ready")
print(f"   Training samples: {train_generator.samples}")
print(f"   Validation samples: {val_generator.samples}")

# ============================================================================
# 2. BUILD MODEL
# ============================================================================
print("\n🏗️ Building VGG16 model...")

# Load pre-trained VGG16
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model layers
base_model.trainable = False
print("   ✓ Frozen VGG16 base layers")

# Add custom top layers
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
print("✅ Model built successfully")

# ============================================================================
# 3. COMPILE MODEL
# ============================================================================
print("\n⚙️ Compiling model...")
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("✅ Model compiled")

# Print model summary
print("\n📋 Model Summary:")
model.summary()

# ============================================================================
# 4. CALLBACKS
# ============================================================================
callbacks = [
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'vgg16_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        verbose=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-7
    )
]

# ============================================================================
# 5. TRAIN MODEL
# ============================================================================
print("\n🚀 Starting training...")
print("=" * 80)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print("=" * 80)
print("✅ Training complete!")

# ============================================================================
# 6. SAVE FINAL MODEL
# ============================================================================
print("\n💾 Saving final model...")
final_model_path = os.path.join(OUTPUT_DIR, 'vgg16.h5')
model.save(final_model_path)
print(f"✅ Final model saved: {final_model_path}")

# ============================================================================
# 7. PLOT TRAINING HISTORY
# ============================================================================
print("\n📈 Creating training history plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
training_plot_path = os.path.join(OUTPUT_DIR, 'training_history.png')
plt.savefig(training_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Training history saved: {training_plot_path}")

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1 COMPLETE! ✅")
print("=" * 80)
print(f"\n📊 Training Summary:")
print(f"   Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"   Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"   Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"   Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"\n💾 Models saved:")
print(f"   Best model: {os.path.join(OUTPUT_DIR, 'vgg16_best.h5')}")
print(f"   Final model: {final_model_path}")
print(f"\n📈 Training plot: {training_plot_path}")
print("\n" + "=" * 80)
print("Ready for STEP 2: Grad-CAM Visualization & Testing!")
print("=" * 80)
