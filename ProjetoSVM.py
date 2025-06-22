import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

dataset_path = 'genres'

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

caracteristicas = []
labels = []

def extrair_caracteristicas(file_path):
    y, sr = librosa.load(file_path, duration=30)  #carrega 30 segundos
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

print("Extraindo características...")

for genero in genres:
    genero_path = os.path.join(dataset_path, genero)
    for file in os.listdir(genero_path):
        file_path = os.path.join(genero_path, file)
        try:
            data = extrair_caracteristicas(file_path)
            caracteristicas.append(data)
            labels.append(genero)
        except Exception as e:
            print(f"Erro no arquivo {file}: {e}")

print("Extração concluída.")

X = np.array(caracteristicas)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Treinando o modelo SVM...")
svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

precisao = accuracy_score(y_test, y_pred)
print(f"Acurácia: {precisao:.2f}")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))