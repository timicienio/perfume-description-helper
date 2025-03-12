# Perfume Description Helper

## 📖 Project Overview / 專案簡介
Perfume Description Helper 是一個使用 **Naïve Bayes** 進行香水文案分類，並利用 **N-gram** 生成文案的語言模型。  
主要功能：
- **文案分類**：對香水文案進行屬性分類。
- **文案生成**：根據使用者選取的屬性、feature 的權重和文案長度，生成符合其風格的香水文案。  
- **前端應用**：使用者可以輸入香水名稱、敘述，獲得香水屬性分類結果，並可自訂香水屬性，調整各 feature 的權重及文案的長度，生成相符的文案。

## 🔧 Tech Stack / 技術使用
- **Model Training / 模型訓練**：Naïve Bayes, N-gram Language Model  
- **Feature Selection / 特徵選取**：Chi-square  
- **Programming Language / 程式語言**：Python  
- **Frontend / 前端展示**：Streamlit

## Demo 影片
- https://drive.google.com/file/d/1Lab-NQxMkePRLCLOPLeS_Z2nJUMo7I3p/view?usp=sharing

## Setting up environment

1. Create conda environment and install requirements.
```
conda create -n perfume-description-helper
conda activate perfume-description-helper
pip install -r requirements.txt
```

2. Download `data` and `dump` from [https://drive.google.com/drive/folders/1w_xbKVFts0GBPH_JeBRUDld8gs-CXyXG?usp=sharing] and copy it to repo root directory (`./data`, `./dump`).
