# Heart Disease Chatbot
## created by Lydia Daids


This is a Streamlit-based chatbot powered by OpenAI + LangChain, designed to explore and analyze a heart disease dataset. This project is a work-in-progress with new features added. Ultimate goal is to incorporate more tuning to make this project more usable/interactive. This project is based on the dataset below, coming from kaggle:
[Dataset:](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

## Features
- Ask natural language questions like:
  - "What does ATA mean?"
  - "What is the average cholesterol for females in their 30s?"
  - "What features influence heart disease?"
- Visual outputs like correlation heatmaps
---

## Instructions

### 1. Clone the repository
git clone https://github.com/LydD20/Heart-Disease-Chatbot.git
cd Heart-Disease-Chatbot

### 2. Create and activate a virtual env
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Add your OpenAI API key
create an .env file in the root directory with :
OPENAI_API_KEY=your_openai_api_key_here

### 5. Run the app
streamlit run app.py

## You now can use the chatbot! 😊


