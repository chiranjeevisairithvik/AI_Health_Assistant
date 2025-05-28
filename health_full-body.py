# STEP 1: PDF Parsing
import pdfplumber
import re
import pandas as pd

# Extract parameters and values from PDF
def extract_parameters_from_pdf(pdf_path):
    parameters = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            lines = text.split('\n')
            for line in lines:
                # Simple regex to match medical parameter patterns
                match = re.match(r"([A-Za-z ()/%+-]+)\s+(\d+[.,]?\d*)\s*([a-zA-Z/]+)?\s*([0-9\-<>.,/]+)?", line)
                if match:
                    param, value, unit, ref_range = match.groups()
                    try:
                        value = float(value.replace(',', ''))
                    except:
                        continue
                    parameters[param.strip()] = {
                        'value': value,
                        'unit': unit,
                        'range': ref_range
                    }
    return parameters

# STEP 2: Normalization

def normalize_parameters(parameters):
    norm_data = []
    for param, details in parameters.items():
        value = details['value']
        ref = details['range']
        if ref and '-' in ref:
            try:
                low, high = map(float, ref.split('-'))
                norm_score = (value - low) / (high - low)
            except:
                norm_score = None
        else:
            norm_score = None
        norm_data.append({
            'parameter': param,
            'value': value,
            'normalized_score': norm_score
        })
    return pd.DataFrame(norm_data)

# STEP 3: Threat Level Prediction (Rule-based)
def threat_level_classifier(df):
    def classify(row):
        score = row['normalized_score']
        if score is None:
            return 'Unknown'
        elif score < 0.8:
            return 'Low'
        elif score < 1.2:
            return 'Normal'
        elif score < 1.5:
            return 'Medium'
        else:
            return 'High'

    df['threat_level'] = df.apply(classify, axis=1)
    return df

# STEP 4: Disease Prediction (Rule-based example)
disease_rules = {
    'Glucose': {'High': 'Diabetes Risk'},
    'Hemoglobin': {'Low': 'Anemia Risk'},
    'WBC': {'High': 'Infection or Inflammation'}
}

def predict_diseases(df):
    disease_risks = []
    for _, row in df.iterrows():
        param = row['parameter']
        level = row['threat_level']
        if param in disease_rules and level in disease_rules[param]:
            disease_risks.append(disease_rules[param][level])
    return list(set(disease_risks))

# STEP 5: Complete Pipeline
def analyze_health_report(pdf_path):
    parameters = extract_parameters_from_pdf(pdf_path)
    df = normalize_parameters(parameters)
    df = threat_level_classifier(df)
    diseases = predict_diseases(df)
    return df, diseases

# Example usage:
# df, diseases = analyze_health_report("example_checkup.pdf")
# print(df)
# print("Predicted Disease Risks:", diseases)
