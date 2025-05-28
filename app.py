import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Set it in your .env file.")

# Load the dataset
df = pd.read_csv("heart.csv")

# Set up the LLM
llm = ChatOpenAI(
    temperature=0.3,
    model_name="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY
)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Safe Python execution tool
python_tool = PythonAstREPLTool(locals={"df": df})

# LangChain agent
agent = initialize_agent(
    tools=[python_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True
)

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Heart Disease EDA Chatbot", layout="wide")
st.title("💉 Heart Disease EDA Chatbot")

st.subheader("Heart Disease Dataset Preview")
st.dataframe(df.head())

# Text input
user_input = st.text_input(
    "Ask a question about the heart disease dataset:",
    "What is the average restingBP for a female in their 40s?"
)

# On Ask
if st.button("Ask"):
    with st.spinner("Generating response..."):
        try:
            user_lower = user_input.lower()
            response = None  # Initialize response string

            # 1. ChestPainType mapping
            # still working on to be more specific
            if re.search(r"\b(chestpaintype|nap|ata|asy|ta)\b", user_lower):
                response = (
                    "ChestPainType mapping:\n"
                    "- TA: Typical Angina\n"
                    "- ATA: Atypical Angina\n"
                    "- NAP: Non-Anginal Pain\n"
                    "- ASY: Asymptomatic"
                )

            # 2. Correlation
            elif "correlation" in user_lower or "influence" in user_lower:
                corr_matrix = df.corr(numeric_only=True)

                # If HeartDisease is mentioned, show correlations and top 3
                if "heartdisease" in user_lower:
                    # Top 3 correlated features (excluding HeartDisease itself)
                    sorted_corr = corr_matrix["HeartDisease"].sort_values(ascending=False)
                    top_3 = sorted_corr.drop("HeartDisease").head(3)
                    st.markdown("### 🔍 Top 3 features most correlated with HeartDisease:")
                    for i, (feature, corr_value) in enumerate(top_3.items(), start=1):
                        st.markdown(f"**{i}. {feature}** — Correlation: `{round(corr_value, 2)}`")

                    sorted_corr = corr_matrix["HeartDisease"].sort_values(ascending=False)
                    st.markdown("### Correlation with HeartDisease")
                    st.dataframe(sorted_corr)

                # Display heatmap last
                st.markdown("### Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
                st.pyplot(fig)

                response = "✅ Correlation table, top 3 features, and heatmap displayed."

            # 3. Lowest cholesterol for someone with heart disease
            elif "lowest cholesterol" in user_lower and "heart disease" in user_lower:
                result = df[df['HeartDisease'] == 1]['Cholesterol'].min()
                response = f"The lowest cholesterol for someone with heart disease is **{round(result, 2)}**."

            # 4. Average restingBP for female in 40s
            elif "average restingbp" in user_lower and "female" in user_lower and ("40s" in user_lower or "40" in user_lower):
                result = df[(df["Sex"] == "F") & (df["Age"].between(40, 49))]["RestingBP"].mean()
                response = f"The average restingBP for a female in their 40s is **{round(result, 2)}**."

            # 5. Average restingBP for male aged 35–40
            elif "average restingbp" in user_lower and "male" in user_lower and ("35" in user_lower and "40" in user_lower):
                result = df[(df["Sex"] == "M") & (df["Age"].between(35, 40))]["RestingBP"].mean()
                response = f"The average restingBP for a male aged between 35 and 40 is **{round(result, 2)}**."

            # 6. Average cholesterol for female in 30s
            elif "average cholesterol" in user_lower and "female" in user_lower and ("30s" in user_lower or "30" in user_lower):
                result = df[(df["Sex"] == "F") & (df["Age"].between(30, 39))]["Cholesterol"].mean()
                response = f"The average cholesterol for a female in their 30s is **{round(result, 2)}**."

            # 7. Fallback to agent
            else:
                query = (
                    "You have a pandas DataFrame `df` with columns like Age, Sex, Cholesterol, RestingBP, and HeartDisease.\n"
                    "Use filters like `df[df['Sex'] == 'F']`, `df['Age'].between(40,49)`, etc.\n"
                    "Always return calculated results using `round()` if applicable.\n\n"
                    f"Question: {user_input}"
                )
                response = agent.run(query)

            # Display chatbot response
            if response:
                st.markdown("### Chatbot Response")
                st.success(response)

        except Exception as e:
            st.error("An error occurred while generating the response.")
            st.exception(e)

# Show chat memory
if memory.buffer:
    st.markdown("### Chat History")
    for i, msg in enumerate(memory.buffer):
        st.markdown(f"**{i+1}.** {msg}")