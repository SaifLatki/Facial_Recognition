# retrain_pipeline.py
import joblib, os, numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# load images (example: walk folders under dataset/lfw-deepfunneled)
images, labels = [], []
root = '../dataset/lfw-deepfunneled/lfw-deepfunneled'
for person in os.listdir(root):
    folder = os.path.join(root, person)
    if not os.path.isdir(folder): continue
    for f in os.listdir(folder):
        if not f.lower().endswith(('.jpg','.png')): continue
        img = Image.open(os.path.join(folder,f)).convert('L').resize((100,100))
        images.append(np.asarray(img).astype(np.float32).ravel())
        labels.append(1 if person=='Arnold_Schwarzenegger' else 0)

X = np.vstack(images)
y = np.array(labels)

scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)

pca = PCA(n_components=150, svd_solver='randomized', whiten=True).fit(Xs)
Xp = pca.transform(Xs)

clf = SVC(kernel='rbf', probability=True).fit(Xp, y)

pipeline = {'scaler': scaler, 'pca': pca, 'svm_model': clf}
joblib.dump(pipeline, 'face_recognition_pipeline.pkl')
print('Saved new face_recognition_pipeline.pkl (100x100 training).')