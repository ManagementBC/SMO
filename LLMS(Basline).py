# === Install Required Packages (for Google Colab) ===
!pip install -q langchain langchain-community langchain-openai langchain-anthropic langchain-google-genai

# === Import Libraries ===
import os
from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

os.environ["OPENAI_API_KEY"] = "XXXXXXXXXXXX"
os.environ["ANTHROPIC_API_KEY"] = "XXXXXXXXXXXX"
os.environ["GOOGLE_API_KEY"] = "XXXXXXXXXXXX"

# === Define the Input Case Prompt ===
cases = {
    "Case 1: Severe Asthma (6y)": """
A 6-year-old child presents with the following:
- Symptoms: Severe cough, wheezing, shortness of breath
- History: Asthma
- SpO₂ Level: 89%
- Heart Rate: 115 bpm

Please provide:
1. Most likely diagnosis
2. Severity classification
3. Treatment recommendations
""",
    "Case 2: Pneumonia in 1y Old (Post-Measles)": """
A 1-year-old malnourished infant presents with the following:
- Symptoms: High fever, cough, rapid breathing
- History: Recent measles infection and malnutrition
- SpO₂ Level: 88%
- Heart Rate: 140 bpm

Please provide:
1. Most likely diagnosis
2. Severity classification
3. Treatment recommendations
""",
    "Case 3: Mild Asthma (9y)": """
A 9-year-old child presents with the following:
- Symptoms: Intermittent wheezing and cough
- History: Asthma
- SpO₂ Level: 96%
- Heart Rate: 95 bpm

Please provide:
1. Most likely diagnosis
2. Severity classification
3. Treatment recommendations
""",
    "Case 4: Elderly Pneumonia (70y)": """
A 70-year-old adult presents with the following:
- Symptoms: Shortness of breath, chest pain, productive cough
- History: Hypertension, Type 2 Diabetes
- SpO₂ Level: 86%
- Heart Rate: 105 bpm

Please provide:
1. Most likely diagnosis
2. Severity classification
3. Treatment recommendations
"""
}

# === Call All Models for Each Case ===
for case_name, prompt in cases.items():
    print(f"\n\n================= {case_name} =================")

    print("\n🔹 GPT-4o:")
    gpt_output = gpt4o.invoke([HumanMessage(content=prompt)]).content
    print(gpt_output)

    print("\n🔹 Claude 3 Opus:")
    claude_output = claude.invoke([HumanMessage(content=prompt)]).content
    print(claude_output)

    print("\n🔹 Gemini 2 Flash:")
    gemini_output = gemini.invoke([HumanMessage(content=prompt)]).content
    print(gemini_output)
