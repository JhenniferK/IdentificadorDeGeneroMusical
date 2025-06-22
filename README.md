# Identificador de Gênero Musical com SVM 🎶🎧

Bem vindo(a)! Esse é um projeto para classificar músicas em diferentes gêneros musicais utilizando Machine Learning, mais especificamente o algoritmo **Support Vector Machine (SVM)**, baseado no dataset **GTZAN**.

## 📋 Sobre o Projeto

Este projeto carrega músicas do dataset GTZAN, extrai características das músicas usando a biblioteca `librosa` (como MFCCs) e treina um modelo SVM para identificar o gênero musical de uma faixa.

## 🗂 Dataset Utilizado

O projeto utiliza o dataset **GTZAN Music Genre Dataset**, que contém 1000 arquivos de áudio divididos em 10 gêneros musicais:

- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

> **Importante:**  
Por questões de tamanho, a pasta `genres/` com o dataset **não está incluída neste repositório**. Você deve baixar o dataset separadamente.

### Link para download do dataset:  
[GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

## 🚀 Como Usar

1. Faça o download do dataset e extraia a pasta `genres/` na raiz do projeto.

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o script principal para treinar e testar o modelo:

```bash
python ProjetoSVM.py
```

## 📦 Dependências

- Python 3.7 ou superior
- numpy
- pandas
- scikit-learn
- librosa
- warnings (já faz parte do Python)

Você pode instalar as dependências com:

```bash
pip install numpy pandas scikit-learn librosa
```

## 🧠 Como Funciona

- O script carrega as músicas por gênero;
- Extrai as características MFCCs de cada música;
- Divide os dados em treino e teste;
- Normaliza os dados;
- Treina o classificador SVM;
- Avalia o desempenho do modelo com acurácia e relatório detalhado.

## 📈 Resultados Esperados

Ao rodar o script, poderá ser visto no terminal:

- O progresso da extração de características;
- Mensagem de treino do modelo;
- Acurácia do modelo na base de teste;
- Relatório de classificação com precisão, recall e F1-score para cada gênero.

---

🖤 **Obrigada por visitar este repositório!**

