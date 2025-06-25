# 📦 Install required tokenizer package
!pip install -q tiktoken

# === 🧠 Imports ===
import pandas as pd
import tiktoken
from IPython.display import display

# === 🔢 Token Counting Utilities ===
enc = tiktoken.encoding_for_model("gpt-4o")

def count_gpt_tokens(text):
    return len(enc.encode(text))

def estimate_tokens_by_char_length(text):
    return len(text) // 4  # Approximate for Claude/Gemini

# === 📋 Your Real Test Cases ===
test_cases = [
    {
        "visit_type": "New",
        "patient_id": "P001",
        "age": 6,
        "weight": 22,
        "symptoms": "Severe cough, wheezing, shortness of breath",
        "spo2": 89,
        "heart_rate": 115,
        "history": "Asthma"
    },
    {
        "visit_type": "New",
        "patient_id": "P002",
        "age": 1,
        "weight": 7.5,
        "symptoms": "High fever, cough, rapid breathing",
        "spo2": 88,
        "heart_rate": 140,
        "history": "Malnutrition, recent measles infection"
    },
    {
        "visit_type": "New",
        "patient_id": "P003",
        "age": 9,
        "weight": 30,
        "symptoms": "Intermittent wheezing and cough",
        "spo2": 96,
        "heart_rate": 95,
        "history": "Asthma"
    },
    {
        "visit_type": "New",
        "patient_id": "P004",
        "age": 70,
        "weight": 68,
        "symptoms": "Shortness of breath, chest pain, productive cough",
        "spo2": 86,
        "heart_rate": 105,
        "history": "Hypertension, Type 2 Diabetes"
    }
]

# === 🧾 Prompt Constructor for Controller ===
def build_prompt(case):
    return f"""
You are a medical agent that can access expert reasoning tools.

Your job is to analyze the following patient case and return your final opinion in this exact 3-point format:

1. Is the condition improving, worsening, or unchanged if it is a follow up? If this is a new case, say: "Not applicable – first recorded visit."
2. Likely Diagnosis
3. Severity Classification
4. Recommended Treatment Plan

== Patient Case ==
Patient ID: {case['patient_id']}
Visit Type: {case['visit_type']}
Age: {case['age']}, Weight: {case['weight']} kg
Symptoms: {case['symptoms']}
SpO₂: {case['spo2']}%, Heart Rate: {case['heart_rate']} bpm
Medical History: {case['history']}
""".strip()

# === 🧪 Run Each Case and Count Tokens ===
results = []

for case in test_cases:
    print(f"🧪 Running Agentic Pipeline for {case['patient_id']}")
    prompt = build_prompt(case)

    # 💬 Controller-level prompt (what GPT-4o sees first)
    controller_input = prompt

    # 🔄 Agentic pipeline function (real function already loaded in your notebook)
    output = run_agentic_combined_rag_synthesis(
        case["visit_type"],
        case["patient_id"],
        case["age"],
        case["weight"],
        case["symptoms"],
        case["spo2"],
        case["heart_rate"],
        case["history"]
    )

    # === 🧠 Token Calculations ===
    results.append({
        "Patient ID": case["patient_id"],
        "Controller Input (GPT-4o)": count_gpt_tokens(controller_input),
        "Controller Output (GPT-4o)": count_gpt_tokens(output),
        "GPT-4o Input": count_gpt_tokens(prompt),
        "GPT-4o Output": count_gpt_tokens(output),
        "Claude Input (est)": estimate_tokens_by_char_length(prompt),
        "Claude Output (est)": estimate_tokens_by_char_length(output),
        "Gemini Input (est)": estimate_tokens_by_char_length(prompt),
        "Gemini Output (est)": estimate_tokens_by_char_length(output),
    })

# === 📊 Show Results in Table ===
df = pd.DataFrame(results)
display(df)


import pandas as pd

# === 🧮 Raw Token Data (you provided) ===
data = [
    {
        "Patient ID": "P001",
        "Controller Input": 144,
        "Controller Output": 190,
        "GPT-4o Input": 144,
        "GPT-4o Output": 190,
        "Claude Input": 145,
        "Claude Output": 234,
        "Gemini Input": 145,
        "Gemini Output": 234
    },
    {
        "Patient ID": "P002",
        "Controller Input": 148,
        "Controller Output": 192,
        "GPT-4o Input": 148,
        "GPT-4o Output": 192,
        "Claude Input": 151,
        "Claude Output": 212,
        "Gemini Input": 151,
        "Gemini Output": 212
    },
    {
        "Patient ID": "P003",
        "Controller Input": 141,
        "Controller Output": 338,
        "GPT-4o Input": 141,
        "GPT-4o Output": 338,
        "Claude Input": 142,
        "Claude Output": 399,
        "Gemini Input": 142,
        "Gemini Output": 399
    },
    {
        "Patient ID": "P004",
        "Controller Input": 149,
        "Controller Output": 202,
        "GPT-4o Input": 149,
        "GPT-4o Output": 202,
        "Claude Input": 153,
        "Claude Output": 280,
        "Gemini Input": 153,
        "Gemini Output": 280
    }
]

df = pd.DataFrame(data)

# === 💵 Pricing Per 1,000 Tokens ===
pricing = {
    "GPT-4o": {"input": 0.005, "output": 0.02},
    "Claude": {"input": 0.015, "output": 0.075},
    "Gemini": {"input": 0.0001, "output": 0.0004}
}

# === 🧮 Cost Calculation Function ===
def calculate_cost(row):
    total_input = row["Controller Input"] + row["GPT-4o Input"] + row["Claude Input"] + row["Gemini Input"]
    total_output = row["Controller Output"] + row["GPT-4o Output"] + row["Claude Output"] + row["Gemini Output"]

    gpt_input_tokens = row["Controller Input"] + row["GPT-4o Input"]
    gpt_output_tokens = row["Controller Output"] + row["GPT-4o Output"]
    claude_input_tokens = row["Claude Input"]
    claude_output_tokens = row["Claude Output"]
    gemini_input_tokens = row["Gemini Input"]
    gemini_output_tokens = row["Gemini Output"]

    gpt_cost = (gpt_input_tokens / 1000) * pricing["GPT-4o"]["input"] + (gpt_output_tokens / 1000) * pricing["GPT-4o"]["output"]
    claude_cost = (claude_input_tokens / 1000) * pricing["Claude"]["input"] + (claude_output_tokens / 1000) * pricing["Claude"]["output"]
    gemini_cost = (gemini_input_tokens / 1000) * pricing["Gemini"]["input"] + (gemini_output_tokens / 1000) * pricing["Gemini"]["output"]

    total_cost = gpt_cost + claude_cost + gemini_cost

    return pd.Series({
        "Total Input Tokens": total_input,
        "Total Output Tokens": total_output,
        "Total Tokens": total_input + total_output,
        "GPT-4o Cost ($)": round(gpt_cost, 5),
        "Claude Cost ($)": round(claude_cost, 5),
        "Gemini Cost ($)": round(gemini_cost, 5),
        "Total Cost ($)": round(total_cost, 5)
    })

# === 📊 Apply and Display Results ===
results = df.join(df.apply(calculate_cost, axis=1))
display(results)

# Optional: Get Total Average
print("\n📈 Average Cost per Visit: $", round(results["Total Cost ($)"].mean(), 5))
print("💰 Total Cost for All Cases:", round(results["Total Cost ($)"].sum(), 5))





