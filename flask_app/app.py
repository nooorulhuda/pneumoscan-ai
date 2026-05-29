from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from datetime import datetime
from fpdf import FPDF
import io

app = Flask(__name__)
app.secret_key = 'pneumonia_fyp_secret_key_2024'

UPLOAD_FOLDER      = 'static/uploads'
GRADCAM_FOLDER     = 'static/gradcam'
MODELS_FOLDER      = r'C:\fyp\h5.files'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER']  = UPLOAD_FOLDER
app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, name, email):
        self.id    = id
        self.name  = name
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if user:
        return User(user['id'], user['name'], user['email'])
    return None

def get_db():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT NOT NULL,
        patient_id TEXT NOT NULL,
        image_path TEXT NOT NULL,
        gradcam_path TEXT,
        model_used TEXT NOT NULL,
        prediction TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp TEXT NOT NULL,
        doctor_id INTEGER NOT NULL)''')
    existing = conn.execute("SELECT * FROM users WHERE email='doctor@fyp.com'").fetchone()
    if not existing:
        hashed = generate_password_hash('doctor123')
        conn.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                     ('Dr. Admin', 'doctor@fyp.com', hashed))
    conn.commit()
    conn.close()

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

MODEL_METRICS = {
    'ResNet50':    {'accuracy': 83.49, 'auc': 0.9713, 'recall': 99.49, 'f1': 88.28, 'precision': 79.35},
    'VGG16':       {'accuracy': 76.12, 'auc': 0.8914, 'recall': 98.97, 'f1': 83.82, 'precision': 72.69},
    'DenseNet121': {'accuracy': 78.85, 'auc': 0.9667, 'recall': 99.74, 'f1': 85.49, 'precision': 74.81},
    'MobileNetV2': {'accuracy': 63.94, 'auc': 0.9600, 'recall': 100.0, 'f1': 77.61, 'precision': 63.41},
    'InceptionV3': {'accuracy': 77.72, 'auc': 0.9458, 'recall': 99.74, 'f1': 84.84, 'precision': 73.81},
}

loaded_models = {}

def build_model(builder, img_size):
    inputs     = keras.Input(shape=(img_size, img_size, 3))
    base_model = builder(input_shape=(img_size, img_size, 3), include_top=False, weights=None)
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs=inputs, outputs=x)

def get_model(model_name):
    if model_name not in loaded_models:
        config       = MODEL_CONFIGS[model_name]
        model        = build_model(config['builder'], config['img_size'])
        weights_path = os.path.join(MODELS_FOLDER, f'{model_name}_best.weights.h5')
        model.load_weights(weights_path)
        loaded_models[model_name] = model
    return loaded_models[model_name]

def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            base_model = layer
            break
    if base_model is None:
        return None
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output])
    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, _ = grad_model(img_tensor)
        loss = tf.reduce_mean(conv_outputs)
    grads        = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0)
    max_val      = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()

def generate_gradcam(image_path, model_name, save_path):
    config    = MODEL_CONFIGS[model_name]
    img_size  = config['img_size']
    model     = get_model(model_name)
    img       = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_size, img_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_proc  = config['preprocess'](img_array.copy())
    img_proc  = np.expand_dims(img_proc, axis=0)
    heatmap   = get_gradcam_heatmap(model, img_proc, config['last_conv'])
    if heatmap is None:
        return None
    img_cv      = cv2.imread(image_path)
    img_cv      = cv2.resize(img_cv, (img_size, img_size))
    heatmap_res = cv2.resize(heatmap, (img_size, img_size))
    heatmap_col = np.uint8(255 * heatmap_res)
    heatmap_col = cv2.applyColorMap(heatmap_col, cv2.COLORMAP_JET)
    overlaid    = (heatmap_col * 0.4 + img_cv * 0.6).astype(np.uint8)
    cv2.imwrite(save_path, overlaid)
    return save_path

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(image_path, model_name):
    config     = MODEL_CONFIGS[model_name]
    img_size   = config['img_size']
    model      = get_model(model_name)
    img        = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_size, img_size))
    img_array  = tf.keras.preprocessing.image.img_to_array(img)
    img_proc   = config['preprocess'](img_array.copy())
    img_proc   = np.expand_dims(img_proc, axis=0)
    pred       = model.predict(img_proc, verbose=0)[0][0]
    label      = 'PNEUMONIA' if pred > 0.5 else 'NORMAL'
    confidence = float(pred) if pred > 0.5 else float(1 - pred)
    return label, round(confidence * 100, 2)

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form['email']
        password = request.form['password']
        conn     = get_db()
        user     = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            u = User(user['id'], user['name'], user['email'])
            login_user(u)
            return redirect(url_for('dashboard'))
        flash('Invalid email or password!', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name     = request.form['name']
        email    = request.form['email']
        password = generate_password_hash(request.form['password'])
        try:
            conn = get_db()
            conn.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                         (name, email, password))
            conn.commit()
            conn.close()
            flash('Account created! Please login.', 'success')
            return redirect(url_for('login'))
        except:
            flash('Email already exists!', 'danger')
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    conn   = get_db()
    total  = conn.execute('SELECT COUNT(*) FROM patients WHERE doctor_id=?', (current_user.id,)).fetchone()[0]
    pneumo = conn.execute("SELECT COUNT(*) FROM patients WHERE doctor_id=? AND prediction='PNEUMONIA'", (current_user.id,)).fetchone()[0]
    normal = conn.execute("SELECT COUNT(*) FROM patients WHERE doctor_id=? AND prediction='NORMAL'", (current_user.id,)).fetchone()[0]
    recent_records = conn.execute('SELECT * FROM patients WHERE doctor_id=? ORDER BY id DESC LIMIT 5', (current_user.id,)).fetchall()
    model_stats = {}
    for m in ['ResNet50', 'VGG16', 'DenseNet121', 'MobileNetV2', 'InceptionV3']:
        count = conn.execute("SELECT COUNT(*) FROM patients WHERE doctor_id=? AND model_used=?", (current_user.id, m)).fetchone()[0]
        model_stats[m] = count
    conn.close()
    return render_template('dashboard.html', total=total, pneumo=pneumo, normal=normal,
                           recent_records=recent_records, model_stats=model_stats)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        patient_name = request.form['patient_name']
        patient_id   = request.form.get('patient_id', 'N/A')
        model_name   = request.form['model_name']
        file         = request.files['xray']

        if file and allowed_file(file.filename):
            filename   = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            label, confidence = predict(image_path, model_name)

            gradcam_filename = f"gradcam_{filename}"
            gradcam_path     = os.path.join(app.config['GRADCAM_FOLDER'], gradcam_filename)
            generate_gradcam(image_path, model_name, gradcam_path)

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn      = get_db()
            conn.execute('''INSERT INTO patients
                (patient_name, patient_id, image_path, gradcam_path,
                 model_used, prediction, confidence, timestamp, doctor_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (patient_name, patient_id,
                 filename,
                 gradcam_filename,
                 model_name, label, confidence, timestamp, current_user.id))
            conn.commit()
            last_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
            conn.close()
            return redirect(url_for('result', record_id=last_id))

        flash('Invalid file! Please upload JPG or PNG.', 'danger')
    return render_template('upload.html', models=list(MODEL_CONFIGS.keys()))

@app.route('/result/<int:record_id>')
@login_required
def result(record_id):
    conn   = get_db()
    record = conn.execute('SELECT * FROM patients WHERE id=?', (record_id,)).fetchone()
    conn.close()
    return render_template('result.html', record=record)

@app.route('/records')
@login_required
def records():
    search = request.args.get('search', '')
    conn   = get_db()
    if search:
        patients = conn.execute(
            "SELECT * FROM patients WHERE doctor_id=? AND (patient_name LIKE ? OR patient_id LIKE ?) ORDER BY id DESC",
            (current_user.id, f'%{search}%', f'%{search}%')).fetchall()
    else:
        patients = conn.execute(
            'SELECT * FROM patients WHERE doctor_id=? ORDER BY id DESC',
            (current_user.id,)).fetchall()
    conn.close()
    return render_template('records.html', patients=patients, search=search)

@app.route('/about')
@login_required
def about():
    return render_template('about.html', metrics=MODEL_METRICS)

@app.route('/download_pdf/<int:record_id>')
@login_required
def download_pdf(record_id):
    conn   = get_db()
    record = conn.execute('SELECT * FROM patients WHERE id=?', (record_id,)).fetchone()
    conn.close()

    img_full_path     = os.path.join(app.config['UPLOAD_FOLDER'],  record['image_path'])
    gradcam_full_path = os.path.join(app.config['GRADCAM_FOLDER'], record['gradcam_path'])

    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(41, 128, 185)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 20)
    pdf.set_xy(0, 8)
    pdf.cell(210, 15, 'Pneumonia Detection Report', ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(236, 240, 241)
    pdf.cell(0, 10, 'Patient Information', ln=True, fill=True)
    pdf.ln(3)
    pdf.set_font('Arial', '', 12)
    pdf.cell(60, 8, 'Patient Name:', border=0)
    pdf.cell(0,  8, record['patient_name'], ln=True, border=0)
    pdf.cell(60, 8, 'Patient ID:', border=0)
    pdf.cell(0,  8, record['patient_id'], ln=True, border=0)
    pdf.cell(60, 8, 'Date & Time:', border=0)
    pdf.cell(0,  8, record['timestamp'], ln=True, border=0)
    pdf.cell(60, 8, 'Model Used:', border=0)
    pdf.cell(0,  8, record['model_used'], ln=True, border=0)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(236, 240, 241)
    pdf.cell(0, 10, 'Diagnosis Result', ln=True, fill=True)
    pdf.ln(3)
    if record['prediction'] == 'PNEUMONIA':
        pdf.set_text_color(192, 57, 43)
    else:
        pdf.set_text_color(39, 174, 96)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 12, f"Result: {record['prediction']}", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Confidence Score: {record['confidence']}%", ln=True, align='C')
    pdf.ln(5)
    if os.path.exists(img_full_path) and os.path.exists(gradcam_full_path):
        pdf.set_font('Arial', 'B', 14)
        pdf.set_fill_color(236, 240, 241)
        pdf.cell(0, 10, 'X-Ray Images', ln=True, fill=True)
        pdf.ln(3)
        pdf.set_font('Arial', '', 10)
        pdf.cell(95, 6, 'Original X-Ray', align='C')
        pdf.cell(95, 6, 'Grad-CAM Heatmap', ln=True, align='C')
        pdf.image(img_full_path,     x=10,  y=None, w=85)
        pdf.set_y(pdf.get_y() - 60)
        pdf.image(gradcam_full_path, x=110, y=None, w=85)
        pdf.ln(5)
    pdf.set_font('Arial', 'I', 9)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 8, 'Disclaimer: This report is AI-generated and should not replace professional medical advice.',
             ln=True, align='C')

    pdf_bytes = bytes(pdf.output())
    pdf_io    = io.BytesIO(pdf_bytes)
    pdf_io.seek(0)
    return send_file(pdf_io, mimetype='application/pdf',
                     as_attachment=True,
                     download_name=f"report_{record['patient_name']}_{record['timestamp'][:10]}.pdf")

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        action = request.form.get('action')
        conn   = get_db()

        if action == 'update_name':
            name = request.form.get('name')
            conn.execute('UPDATE users SET name=? WHERE id=?', (name, current_user.id))
            conn.commit()
            flash('Name updated successfully!', 'success')

        elif action == 'update_email':
            email = request.form.get('email')
            try:
                conn.execute('UPDATE users SET email=? WHERE id=?', (email, current_user.id))
                conn.commit()
                flash('Email updated successfully!', 'success')
            except:
                flash('Email already exists!', 'danger')

        elif action == 'update_password':
            old_password = request.form.get('old_password')
            new_password = request.form.get('new_password')
            user = conn.execute('SELECT * FROM users WHERE id=?', (current_user.id,)).fetchone()
            if check_password_hash(user['password'], old_password):
                conn.execute('UPDATE users SET password=? WHERE id=?',
                             (generate_password_hash(new_password), current_user.id))
                conn.commit()
                flash('Password updated successfully!', 'success')
            else:
                flash('Old password is incorrect!', 'danger')

        conn.close()
        return redirect(url_for('profile'))

    return render_template('profile.html')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)