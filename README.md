# 🎓 Student Dropout Prediction and Support System

A smart web application designed to help educational institutions identify students at risk of dropping out — using data-driven machine learning models, insightful visualizations, and an intelligent chatbot.

---

## 🚀 Overview

Educational institutions often struggle to proactively detect students likely to drop out due to academic or behavioral challenges. This system empowers schools with a predictive model and tools to:

- Analyze student data
- Predict dropout risk
- Provide intervention guidance
- Assist users via a built-in AI chatbot (EduBot)

---

## 🧠 Key Features

✅ **Dropout Prediction (ML-powered)**  
✅ **Data Visualization** – Explore trends by gender, alcohol use, grades, etc.  
✅ **Smart Chatbot (EduBot)** – Built using LLaMA-4 via Groq API  
✅ **PDF Report Generator** – Download prediction results  
✅ **Clean UI** – Built with Streamlit  

---

## 🛠️ Tech Stack

| Category         | Tools Used                              |
|------------------|------------------------------------------|
| Frontend         | Streamlit                                |
| ML Model         | Random Forest Classifier (sklearn)       |
| Data Handling    | pandas, seaborn, matplotlib               |
| Chatbot Engine   | LangChain + Groq (LLaMA-4)               |
| PDF Reporting    | fpdf                                     |
| Encoding         | LabelEncoder, SMOTE for class imbalance  |

---

## 🗂️ Project Structure

📁 edubot/
│ ├── engine.py # LLM-powered chatbot
│ └── knowledge/ # Markdown files for EduBot
│ ├── model_info.md
│ ├── dropout_causes.md
│ ├── app_guide.md
│ ├── parent_guidance.md
│ ├── feature_explanation.md
│ └── faq.md
├── reduced_student_dropout.csv
├── model.pkl
├── random.pkl
├── app.py
└── README.md


---

## 🧪 How to Run Locally

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/dropout-predictor.git
cd dropout-predictor
```
2. **Create virtual environment (optional)**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Add your .env file(for Groq)**
```bash
GROQ_API_KEY=your_actual_key_here
```
5. **Run the app**
```bash
streamlit run app.py
```


