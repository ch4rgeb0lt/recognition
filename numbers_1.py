import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# CHECK FOR EXISTING MODEL 
model_file = "model.h5"
label_encoder = LabelEncoder()

if os.path.exists(model_file):
    print(f"Found existing model: {model_file}")
    print("Loading model...")
    model = tf.keras.models.load_model(model_file)
    
    # Load label encoder from CSV
    df = pd.read_csv("train.csv")
    label_encoder.fit_transform(df['Font'])
    print("Model loaded successfully!")
    
else:
    print(f"No existing model found. Training new model...")
    
    # LOAD DATA 
    print("Loading data from train.csv...")
    df = pd.read_csv("train.csv")

    # Encode labels (A-Z, a-z, 0-9 = 62 classes)
    label_encoder.fit_transform(df['Font'])

    # LOAD IMAGES 
    print("Loading images from Font folder...")
    images = []
    labels = []

    for idx, row in df.iterrows():
        # Get filename and folder from path
        path_parts = row['filepaths'].split('/')
        filename = path_parts[-1]
        folder = path_parts[-2]
        
        img_path = os.path.join("Font", folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            img = cv2.resize(img, (28, 28)) / 255.0
            images.append(img)
            labels.append(row['label'])
        
        # Show progress
        if (idx + 1) % 10000 == 0:
            print(f"  Loaded {idx + 1} images...")

    X = np.array(images).reshape(-1, 28, 28, 1)
    y = np.array(labels)
    print(f"Loaded {len(X)} images successfully!")

    # TRAIN MODEL 
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(62, activation='softmax')
    ])

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    model.fit(X_train, y_train, 
              epochs=5, 
              validation_data=(X_test, y_test), 
              verbose=1)

    # SAVE MODEL
    model.save(model_file)
    print(f"Model saved as {model_file}!")

# LIVE CAMERA
print("\n" + "="*50)
print("Starting camera - Press 'q' to quit")
print("="*50)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Draw box
    cv2.rectangle(frame, (200,100), (400,300), (0,255,0), 2)
    
    # Process ROI
    roi = frame[100:300, 200:400]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (28,28)) / 255.0
    img = img.reshape(1,28,28,1)
    
    # Predict
    pred = model.predict(img, verbose=0)
    letter = label_encoder.inverse_transform([np.argmax(pred)])[0]
    confidence = np.max(pred) * 100
    
    # Show result
    cv2.putText(frame, f"Pred: {letter} ({confidence:.1f}%)", (50,50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Character Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("\nCamera closed. Goodbye!")