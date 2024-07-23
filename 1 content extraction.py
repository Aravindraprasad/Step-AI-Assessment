import PyPDF2
import re
import os

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)

        full_text = ""
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text = page.extract_text()
            text = re.sub(r'\s+', ' ', text)  
            text = text.strip()

            full_text += text + "\n\n"  

    return full_text

def save_text_to_file(text, output_path):

    cleaned_text = text.encode('utf-8', 'ignore').decode('utf-8')
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

def process_textbooks(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    combined_text = ""

    for filename in os.listdir(input_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            print(f"Processing textbook: {filename}")

            extracted_text = extract_text_from_pdf(pdf_path)

            combined_text += f"--- START OF {filename} ---\n\n"
            combined_text += extracted_text
            combined_text += f"\n\n--- END OF {filename} ---\n\n"

    output_path = os.path.join(output_folder, "combined_textbooks.txt")
    save_text_to_file(combined_text, output_path)
    print(f"Combined text from all textbooks saved to {output_path}")

    print("All textbooks processed and combined.")


input_folder = r"./Books"
output_folder = r"./processed_textbooks"

process_textbooks(input_folder, output_folder)