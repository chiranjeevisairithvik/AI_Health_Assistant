import fitz  # PyMuPDF
import json
import os

# Folder where PDFs are stored
pdf_folder = "Health_pdf"

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_folder):
    medical_data = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
            
            medical_data.append({"file_name": pdf_file, "content": text})
    
    return medical_data

# Extract data and save as JSON
medical_data = extract_text_from_pdfs(pdf_folder)
with open("medical_data.json", "w") as json_file:
    json.dump(medical_data, json_file, indent=4)

print("✅ Medical data extracted and saved as `medical_data.json`.")
