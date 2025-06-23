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
input_case = """
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

Please provide:
1. Most likely diagnosis
2. Severity classification
3. Treatment recommendations
"""

# === Initialize LLMs ===
gpt4o = ChatOpenAI(model="gpt-4o", temperature=0.2)
claude = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.2)
gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# === Call All 3 LLMs (Without RAG) ===
responses = {}

print("🔹 Running GPT-4o...")
responses["GPT-4o"] = gpt4o.invoke([HumanMessage(content=input_case)]).content

print("🔹 Running Claude 3 Opus...")
responses["Claude 3"] = claude.invoke([HumanMessage(content=input_case)]).content

print("🔹 Running Gemini 2 Flash...")
responses["Gemini"] = gemini.invoke([HumanMessage(content=input_case)]).content

# === Display Results ===
for model, output in responses.items():
    print(f"\n=== {model} ===")
    print(output)
