"""
Numerical Consistency Experiment
Models: GPT-4, Claude Sonnet 3.5, Gemini, Llama 3
Measures consistency across unit conversions and checks plausibility.
"""

import csv
import re
import time
from typing import Dict, List, Tuple, Optional
import pandas as pd
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
# For Llama 3, you might need a specific client; adjust accordingly.
# Here we use a placeholder.

# ==================== Configuration ====================
# Insert your API keys (or set environment variables)
OPENAI_API_KEY = "your-openai-key"
ANTHROPIC_API_KEY = "your-anthropic-key"
GOOGLE_API_KEY = "your-google-key"
# Llama 3 (if using a provider like Replicate, Groq, etc.)
LLAMA_API_KEY = "your-llama-key"
LLAMA_ENDPOINT = "https://api.groq.com/openai/v1"  # example

# Model names
MODELS = {
    "gpt4": "gpt-4",
    "claude": "claude-3-sonnet-20241022",  # adjust
    "gemini": "gemini-1.5-pro",
    "llama3": "llama3-70b-8192"
}

# ==================== Helper Functions ====================
def extract_number(text: str) -> Optional[float]:
    """Extract the first numeric value (including decimals) from a string."""
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None

def convert(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert between common units.
    Supports meters/feet/inches, kilograms/pounds, liters/gallons.
    Returns value in `to_unit`.
    """
    # Length conversions
    if from_unit in ["meters", "m"] and to_unit in ["feet", "ft"]:
        return value * 3.28084
    if from_unit in ["feet", "ft"] and to_unit in ["meters", "m"]:
        return value / 3.28084
    if from_unit in ["meters", "m"] and to_unit in ["inches", "in"]:
        return value * 39.3701
    if from_unit in ["inches", "in"] and to_unit in ["meters", "m"]:
        return value / 39.3701
    if from_unit in ["feet", "ft"] and to_unit in ["inches", "in"]:
        return value * 12
    if from_unit in ["inches", "in"] and to_unit in ["feet", "ft"]:
        return value / 12

    # Weight conversions (kg <-> lb)
    if from_unit in ["kg", "kilograms"] and to_unit in ["lb", "pounds"]:
        return value * 2.20462
    if from_unit in ["lb", "pounds"] and to_unit in ["kg", "kilograms"]:
        return value / 2.20462

    # Volume conversions (liter <-> gallon)
    if from_unit in ["L", "liters"] and to_unit in ["gal", "gallons"]:
        return value * 0.264172
    if from_unit in ["gal", "gallons"] and to_unit in ["L", "liters"]:
        return value / 0.264172

    # Same unit -> no change
    if from_unit == to_unit:
        return value

    raise ValueError(f"Unsupported conversion: {from_unit} -> {to_unit}")

def is_within_range(value_si: float, min_si: float, max_si: float) -> bool:
    """Check if value in SI units is within plausible range."""
    return min_si <= value_si <= max_si

# ==================== Model Query Functions ====================
def query_gpt4(prompt: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=MODELS["gpt4"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=50
    )
    return response.choices[0].message.content.strip()

def query_claude(prompt: str) -> str:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=MODELS["claude"],
        max_tokens=50,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

def query_gemini(prompt: str) -> str:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(MODELS["gemini"])
    response = model.generate_content(prompt, generation_config={"temperature": 0.0})
    return response.text.strip()

def query_llama(prompt: str) -> str:
    # Example using Groq API (OpenAI‑compatible)
    import openai
    client = openai.OpenAI(api_key=LLAMA_API_KEY, base_url=LLAMA_ENDPOINT)
    response = client.chat.completions.create(
        model=MODELS["llama3"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=50
    )
    return response.choices[0].message.content.strip()

# ==================== Main Experiment ====================
def run_experiment(dataset_path: str, output_path: str):
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Prepare results storage
    results = []

    # For each model
    for model_name, query_func in [
        ("gpt4", query_gpt4),
        ("claude", query_claude),
        ("gemini", query_gemini),
        ("llama3", query_llama)
    ]:
        print(f"\nRunning {model_name}...")

        for idx, row in df.iterrows():
            object_name = row["object"]
            question = row["question_text"]
            expected_units = row["expected_units"]
            min_si = row["plausible_min"]
            max_si = row["plausible_max"]

            # Query model
            try:
                answer = query_func(question)
            except Exception as e:
                print(f"Error querying {model_name} for question {idx}: {e}")
                answer = None

            # Parse number
            value = extract_number(answer) if answer else None

            # Record raw answer and parsed value
            results.append({
                "model": model_name,
                "object": object_name,
                "question": question,
                "expected_units": expected_units,
                "raw_answer": answer,
                "parsed_value": value,
                "min_si": min_si,
                "max_si": max_si
            })

            # Be kind to APIs
            time.sleep(1)

        # After each model, we can compute within‑object consistency later
        # For now, save raw results

    # Save raw results
    raw_df = pd.DataFrame(results)
    raw_df.to_csv(output_path.replace(".csv", "_raw.csv"), index=False)
    print("Raw results saved.")

    # Now process consistency and plausibility
    # (You will need to pair questions per object and unit variant)
    # This is a simplified example; you'll need to expand it.

    # For demonstration, we group by model and object, then compare metric vs imperial answers.
    # We'll assume we can identify which question is metric and which is imperial based on expected_units.
    # We'll do a second pass.

    consistency_records = []
    for model_name in raw_df["model"].unique():
        model_data = raw_df[raw_df["model"] == model_name]
        for obj in model_data["object"].unique():
            obj_data = model_data[model_data["object"] == obj]
            # Find metric and imperial answers for the same quantity type
            metric_rows = obj_data[obj_data["expected_units"].isin(["meters", "kg", "liters"])]
            imperial_rows = obj_data[obj_data["expected_units"].isin(["feet", "inches", "pounds", "gallons"])]

            if len(metric_rows) == 0 or len(imperial_rows) == 0:
                continue

            # For each metric and imperial pair (simplified: take the first of each)
            for _, m_row in metric_rows.iterrows():
                for _, i_row in imperial_rows.iterrows():
                    if m_row["parsed_value"] is None or i_row["parsed_value"] is None:
                        continue
                    # Convert imperial to metric (SI) for comparison
                    from_unit = i_row["expected_units"]
                    to_unit = m_row["expected_units"]  # we convert to the metric unit
                    try:
                        imperial_si = convert(i_row["parsed_value"], from_unit, to_unit)
                    except ValueError as e:
                        print(f"Skipping conversion: {e}")
                        continue
                    metric_val = m_row["parsed_value"]

                    # Relative Consistency Score
                    avg = (metric_val + imperial_si) / 2
                    diff = abs(metric_val - imperial_si)
                    rcs = 1 - (diff / avg) if avg != 0 else 0
                    # Cap at 0
                    rcs = max(0, rcs)

                    # Plausibility check
                    metric_plausible = is_within_range(metric_val, m_row["min_si"], m_row["max_si"])
                    imperial_plausible = is_within_range(imperial_si, m_row["min_si"], m_row["max_si"])

                    consistency_records.append({
                        "model": model_name,
                        "object": obj,
                        "metric_question": m_row["question"],
                        "imperial_question": i_row["question"],
                        "metric_value": metric_val,
                        "imperial_raw": i_row["parsed_value"],
                        "imperial_converted": imperial_si,
                        "rcs": rcs,
                        "metric_plausible": metric_plausible,
                        "imperial_plausible": imperial_plausible,
                        "metric_unit": m_row["expected_units"],
                        "imperial_unit": i_row["expected_units"]
                    })

    cons_df = pd.DataFrame(consistency_records)
    cons_df.to_csv(output_path.replace(".csv", "_consistency.csv"), index=False)
    print("Consistency results saved.")
    print("\nSummary (mean RCS per model):")
    print(cons_df.groupby("model")["rcs"].mean())

if __name__ == "__main__":
    # Replace with the actual path to your dataset CSV file
    dataset_file = "pqcb_dataset.csv"
    output_file = "experiment_results.csv"
    run_experiment(dataset_file, output_file)