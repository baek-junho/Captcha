import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 전처리 함수 ---
def preprocess_image(img):
    # 블러 → 이진화 → 노이즈 제거 → 중심 정렬 → resize
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphology (노이즈 제거)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # 문자 중심으로 자르기
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img[y:y+h, x:x+w]
    else:
        cropped = img

    # 정사각형 패딩
    h, w = cropped.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off+h, x_off:x_off+w] = cropped

    resized = cv2.resize(padded, (28, 28))
    return resized / 255.0

# --- 학습 데이터 로딩 ---
# X, y = [], []
DATA_DIR = './image/labeled'

X, y = [], []
for fname in os.listdir(DATA_DIR):
    if not fname.endswith('.jpg'):
        continue

    label_part = fname.split('_')[-1].replace('.jpg', '')
    if len(label_part) != 4:
        continue

    img_path = os.path.join(DATA_DIR, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    h, w = img.shape
    char_w = w // 4

    for i in range(4):
        char_img = img[:, i * char_w:(i + 1) * char_w]
        processed = preprocess_image(char_img)
        flattened = processed.flatten()
        X.append(flattened)
        y.append(label_part[i])

print(f"총 학습 문자 수: {len(X)}개")

# --- 모델 학습 ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 평가 ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"테스트 정확도: {acc:.4f}")



#################


import cv2
import numpy as np

def predict_image(filename, model):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("이미지 없음")
        return ""

    h, w = img.shape
    char_w = w // 4
    result = ""

    for i in range(4):
        char_img = img[:, i * char_w:(i + 1) * char_w]
        processed = preprocess_image(char_img)
        flattened = processed.flatten()
        pred = model.predict([flattened])[0]
        result += pred
    return result

# 사용 예시
print("예측 결과:", predict_image('./image/hb/test2.jpg', model))


import joblib

# 저장
joblib.dump(model, './image/rf_captcha_model.pkl')