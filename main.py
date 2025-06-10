from flask import Flask, render_template, request, jsonify
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
import fitz

load_dotenv(find_dotenv(), override=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# Key points model
#class Review(BaseModel):
 #   key_points: list[str] = Field(description="Extract top 5 key terms (keywords or short phrases) from the text.")

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def generate_key_points(text):
    prompt = (
        "Extract the top 5 key terms (keywords or short phrases) from the following content:\n\n"
        f"{text}\n\nOnly return the list of terms separated by commas."
    )
    response = llm.invoke(prompt)
    key_terms = [term.strip() for term in response.content.split(",") if term.strip()]
    return key_terms


def process_uploaded_input(input_type, request, key_points_accumulator):
    input_text = ""
    if request.form.get(f'{input_type}_type') == 'pdf':
        file = request.files.get(f'{input_type}_pdf')
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            input_text = extract_text_from_pdf(filepath)
    else:
        input_text = request.form.get(f'{input_type}_text', '')

    if input_text:
        key_points = generate_key_points(input_text)
        key_points_accumulator.extend(key_points)
        return {f'{input_type}_text': input_text, f'{input_type}_key_points': key_points}
    return {}

@app.route('/', methods=['GET', 'POST'])
def index():
    data = {}
    all_key_points = []

    if request.method == 'POST':
        # Syllabus
        if request.form.get('syllabus_enable'):
            data.update(process_uploaded_input('syllabus', request, all_key_points))

        # Notes
        if request.form.get('notes_enable'):
            data.update(process_uploaded_input('notes', request, all_key_points))

        # PYQs
        if request.form.get('pyqs_enable'):
            data.update(process_uploaded_input('pyqs', request, all_key_points))

        # Unique key terms for display
        unique_tags = list(set(all_key_points))

        # Weightage Tags
        selected_tags = request.form.getlist('weightage_tags')
        data['weightage_tags'] = selected_tags

        # Sections
        sections = []
        sec_count = int(request.form.get('sections_count', 0))
        for i in range(1, sec_count + 1):
            sections.append({
                'type': request.form.get(f'section_{i}_type'),
                'count': request.form.get(f'section_{i}_count'),
                'marks': request.form.get(f'section_{i}_marks')
            })
        data['sections'] = sections

        print("Collected User Input Data:")
        print(data)

        return render_template('form.html', weightage_tags_list=unique_tags)

    return render_template('form.html', weightage_tags_list=[])

@app.route('/extract_key_terms', methods=['POST'])
def extract_key_terms_api():
    text = request.form.get('text', '')
    if not text:
        return jsonify({'key_terms': []})
    try:
        key_terms = generate_key_points(text)
        return jsonify({'key_terms': key_terms})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 