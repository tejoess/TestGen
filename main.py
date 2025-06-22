from flask import Flask, render_template, request, jsonify, session, redirect
from flask_session import Session
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
import fitz  # PyMuPDF
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import json
import traceback

load_dotenv(find_dotenv(), override=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'testgen-secret-key-2024'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False

Session(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.0)

#--------------------------------INPUTS--------------------------------
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

#--------------------------------QUESTION GENERATION--------------------------------
def text_splitting(text):
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    return splitter.split_text(text)

def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    documents = [Document(page_content=chunk) for chunk in chunks]
    return FAISS.from_documents(documents, embeddings)

def create_retriever(vector_store, query):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever.invoke(query)

def build_user_query(data: dict) -> str:
    query_parts = []

    # Add topic-based summaries (if available)
    if "syllabus_key_points" in data and data["syllabus_key_points"]:
        query_parts.append(f"Syllabus topics: {', '.join(data['syllabus_key_points'])}")
    
    if "notes_key_points" in data and data["notes_key_points"]:
        query_parts.append(f"Notes highlights: {', '.join(data['notes_key_points'])}")

    if "weightage_tags" in data and data["weightage_tags"]:
        query_parts.append(f"Focus areas: {', '.join(data['weightage_tags'])}")

    # Construct section-specific instructions
    if "sections" in data and data["sections"]:
        section_descriptions = []
        for idx, section in enumerate(data["sections"], start=1):
            section_descriptions.append(
                f"Section {idx}: Generate {section['count']} '{section['type']}' type questions of {section['marks']} marks each"
            )
        query_parts.append("\n".join(section_descriptions))

    # Combine all parts
    user_question = "\n".join(query_parts)
    return user_question

def generate_questions(context, user_query, sections):
    # Create dynamic section structure based on actual sections
    section_structure = {}
    for i, section in enumerate(sections, 1):
        section_structure[f"Section {i}"] = {
            "questions": [
                {"question_no": j, "question": "question text", "marks": section['marks']}
                for j in range(1, section['count'] + 1)
            ]
        }
    
    # Convert to JSON string for the prompt
    structure_example = json.dumps(section_structure, indent=2)
    
    prompt = PromptTemplate(
        template = """
        You are a question paper generator AI.

        Consider the following context which includes syllabus content, notes, and previous year questions. Generate questions based on this comprehensive content and the instructions provided.

        Each question must:
        - Be relevant to the provided content (syllabus/notes/PYQs)
        - Match the difficulty and depth according to the marks assigned (e.g., 2 marks = objective/basic, 10 marks = detailed/conceptual)
        - Be non-repetitive and well-structured
        - Avoid subtopics in output
        - Draw from the most important concepts in the provided content

        Provide the output in Python dictionary/JSON format only, structured exactly like this:

        {structure_example}

        Context (Combined Syllabus, Notes, and PYQs):
        {context}

        Instructions:
        {question}

        If context is missing or insufficient, just say: "Insufficient content available". Do not provide partial or inaccurate responses.
        """,
        input_variables=["context", "question", "structure_example"]
        )
    chain = prompt | llm | JsonOutputParser()
    return chain.invoke({"context": context, "question": user_query, "structure_example": structure_example})

def generate_question_paper(data):
    """Main function to generate question paper from collected data"""
    try:
        # Combine all available text content
        all_text_content = []
        
        # Add syllabus text
        if data.get('syllabus_text'):
            all_text_content.append(f"SYLLABUS CONTENT:\n{data['syllabus_text']}")
        
        # Add notes text
        if data.get('notes_text'):
            all_text_content.append(f"NOTES CONTENT:\n{data['notes_text']}")
        
        # Add PYQs text
        if data.get('pyqs_text'):
            all_text_content.append(f"PREVIOUS YEAR QUESTIONS:\n{data['pyqs_text']}")
        
        # Combine all text
        combined_text = "\n\n".join(all_text_content)
        
        if not combined_text:
            return {"error": "No content available for question generation. Please provide syllabus, notes, or PYQs."}
        
        print(f"DEBUG: Combined text length: {len(combined_text)}")
        print(f"DEBUG: Text sources: {[key for key in ['syllabus_text', 'notes_text', 'pyqs_text'] if data.get(key)]}")
        
        # Split text into chunks
        chunks = text_splitting(combined_text)
        if not chunks:
            return {"error": "Could not process combined text content"}
        
        # Create vector store (this will be reused for evaluation)
        vector_store = create_vector_store(chunks)
        
        # Build user query
        user_query = build_user_query(data)
        
        # Get relevant context
        context = create_retriever(vector_store, user_query)
        
        # Generate questions
        result = generate_questions(context, user_query, data['sections'])
        
        # Store document chunks in session for evaluation
        session['document_chunks'] = chunks
        session['combined_text'] = combined_text
        
        print(f"DEBUG: Stored {len(chunks)} document chunks in server-side session")
        print(f"DEBUG: Session keys: {list(session.keys())}")
        
        return result
    except Exception as e:
        return {"error": f"Error generating questions: {str(e)}"}

@app.route('/', methods=['GET', 'POST'])
def index():
    # Clear session and uploaded files on a new start
    if request.method == 'GET':
        session.clear()
        # Optional: Clear the uploads folder as well
        # for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        #     os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("DEBUG: Session and uploads cleared for new run.")

    data = {}
    all_key_points = []

    if request.method == 'POST':
        # Handle file uploads and key terms extraction immediately
        for input_type in ['syllabus', 'notes', 'pyqs']:
            if request.form.get(f'{input_type}_enable'):
                # Check if we have uploaded files in session
                uploaded_files = session.get('uploaded_files', {})
                if input_type in uploaded_files:
                    # Use the uploaded file data
                    file_data = uploaded_files[input_type]
                    data[f'{input_type}_text'] = file_data['text']
                    data[f'{input_type}_key_points'] = file_data['key_points']
                    all_key_points.extend(file_data['key_points'])
                else:
                    # Process current form data
                    result = process_uploaded_input(input_type, request, all_key_points)
                    data.update(result)

        # Unique key terms for display
        unique_tags = list(set(all_key_points))

        # Weightage Tags
        selected_tags = request.form.getlist('weightage_tags')
        data['weightage_tags'] = selected_tags

        # Sections
        sections_count = int(request.form.get('sections_count', 0))
        sections = []
        for i in range(1, sections_count + 1):
            section = {
                'type': request.form.get(f'section_{i}_type', ''),
                'count': int(request.form.get(f'section_{i}_count', 0)),
                'marks': int(request.form.get(f'section_{i}_marks', 0))
            }
            sections.append(section)
        data['sections'] = sections

        print("Collected User Input Data:")
        print(data)

        # Generate question paper
        question_paper = generate_question_paper(data)
        
        # Store data in session for result page
        session['input_data'] = data
        session['question_paper'] = question_paper
        session.modified = True
        
        print(f"DEBUG: After question generation - Session keys: {list(session.keys())}")
        print(f"DEBUG: Document chunks in session: {'document_chunks' in session}")

        return render_template('result.html', question_paper=question_paper, input_data=data)
    
    return render_template('form.html', weightage_tags_list=[])

@app.route('/extract_key_terms', methods=['POST'])
def extract_key_terms_api():
    text = request.form.get('text', '')
    input_type = request.form.get('input_type', '')
    
    if not text:
        return jsonify({'key_terms': []})
    
    try:
        key_terms = generate_key_points(text)
        
        # Store in session for form submission
        uploaded_files = session.get('uploaded_files', {})
        uploaded_files[input_type] = {
            'text': text,
            'key_points': key_terms,
            'filepath': None
        }
        session['uploaded_files'] = uploaded_files
        
        return jsonify({'key_terms': key_terms})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    input_type = request.form.get('input_type', '')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Extract text and generate key points
        text = extract_text_from_pdf(filepath)
        key_points = generate_key_points(text)
        
        # Store in session for form submission
        uploaded_files = session.get('uploaded_files', {})
        uploaded_files[input_type] = {
            'text': text,
            'key_points': key_points,
            'filepath': filepath
        }
        session['uploaded_files'] = uploaded_files
        
        return jsonify({
            'status': 'success',
            'key_points': key_points,
            'input_type': input_type,
            'text': text[:500] + "..." if len(text) > 500 else text  # Truncate for display
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/submit_answers', methods=['POST'])
def submit_answers():
    try:
        print("DEBUG: ========== SUBMIT ANSWERS STARTED ==========")
        print(f"DEBUG: Session keys at start: {list(session.keys())}")
        
        data = request.get_json()
        print("DEBUG: Received data:", data)
        
        answers = data.get('answers', {})
        question_paper = data.get('question_paper', {})
        
        print("DEBUG: Answers:", answers)
        print("DEBUG: Question paper keys:", list(question_paper.keys()))
        
        # Create answersheet dictionary
        answersheet = {}
        
        # We need to filter out the internal keys I added before from the question paper
        sections_to_process = {k: v for k, v in question_paper.items() if not k.startswith('_')}

        for section_name, section_data in sections_to_process.items():
            print(f"DEBUG: Processing section {section_name}")
            if section_name in answers:
                print(f"DEBUG: Found answers for {section_name}:", answers[section_name])
                answersheet[section_name] = {
                    'questions': []
                }
                
                for question in section_data.get('questions', []):
                    question_no = question['question_no']
                    question_no_str = str(question_no)
                    student_answer = answers[section_name].get(question_no_str, {}).get('answer', '')
                    print(f"DEBUG: Question {question_no}, Answer: '{student_answer}'")
                    
                    answersheet[section_name]['questions'].append({
                        'question_no': question_no,
                        'question': question['question'],
                        'marks': question['marks'],
                        'student_answer': student_answer
                    })
            else:
                print(f"DEBUG: No answers found for section {section_name}")
        
        print("\n" + "="*50)
        print("ANSWERSHEET DICTIONARY:")
        print("="*50)
        print(json.dumps(answersheet, indent=2))
        print("="*50 + "\n")
        
        # Get document chunks from session for evaluation
        document_chunks = session.get('document_chunks')
        print(f"DEBUG: Retrieved document_chunks from session: {document_chunks is not None}")
        
        if document_chunks:
            print(f"DEBUG: Number of chunks available: {len(document_chunks)}")
            try:
                print("DEBUG: Starting enrichment process...")
                enriched_answersheet = enrich_answersheet_with_context(answersheet, document_chunks)
                print("DEBUG: Enrichment completed successfully")
                
                print("DEBUG: Starting evaluation process...")
                eval_report = evaluate_answers(enriched_answersheet)
                print("DEBUG: Evaluation completed successfully")
                print(f"DEBUG: Evaluation report: {eval_report}")
                
                session['eval_report'] = eval_report
                session['answersheet'] = answersheet
                
                return jsonify({
                    'success': True,
                    'answersheet': answersheet,
                    'redirect': '/report'
                })
            except Exception as eval_error:
                print(f"DEBUG: Error during evaluation process: {str(eval_error)}")
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': f'Evaluation error: {str(eval_error)}'
                }), 500
        else:
            return jsonify({
                'success': False,
                'error': 'Document chunks not found in session. Please regenerate the question paper.'
            }), 500
        
    except Exception as e:
        print(f"DEBUG: Error in submit_answers: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/debug_session')
def debug_session():
    """Debug route to check session data"""
    session_data = {
        'keys': list(session.keys()),
        'document_chunks_count': len(session.get('document_chunks', [])),
        'has_question_paper': 'question_paper' in session,
        'has_uploaded_files': 'uploaded_files' in session
    }
    return jsonify(session_data)

@app.route('/report')
def report():
    eval_report = session.get('eval_report')
    answersheet = session.get('answersheet')
    
    if not eval_report or not answersheet:
        return redirect('/')
    
    return render_template('report.html', eval_report=eval_report, answersheet=answersheet)


#--------------------------------EVALUATION--------------------------------

def evaluate_answers(answersheet):
    prompt = PromptTemplate(
        template = """
        You are an expert answer evaluator. You are provided with a structured answer sheet where each question has:
        - the original question
        - the student's answer
        - the number of marks assigned
        - relevant context extracted from notes/syllabus

        Your task is to evaluate each answer based on:
        - Relevance to the context provided
        - Completeness, correctness, and clarity of the response
        - Technical accuracy based on standard knowledge

        ### Evaluation Requirements:

        1. **For each question**, do the following:
        - Evaluate the student's answer.
        - Assign marks out of the given marks (fractional marks are allowed but only in terms of 0.5 and .0).
        
        2. After all evaluations, generate:
        - total_score out of total_possible_marks
        - 3-4 Areas of Improvement (in detailed sentences)
        - 3-4 Suggestions (tips or practices to improve)
        - 4-5 Weak Topics as specific technical terms or domain-specific concepts (e.g., "Machine Learning applications", "Database Transactions", "TCP/IP Layers"). Avoid vague or general skills like "clarity","concise writing", or "understanding the question".
        These weak topics will be used to recommend learning resources, so they must be study topics only, derived either from the question or the provided context.

        ### Output Format (Python dictionary/JSON):
        {{
        "score": "X / Y",
        "improvement_areas": [
            "The student struggles to articulate precise advantages of certain technologies."
        ],
        "suggestions": [
            "Revise core concepts using summary notes or flashcards."
        ],
        "weak_topics": ["Machine learning applications", "AI Agents","Python asyncio","PCA component"]
        }}
        
        Use ONLY the context provided to validate correctness, but if the student's answer is independently correct based on general knowledge, still reward them.

        Avoid giving marks for vague or incorrect answers. Be strict but fair.

        Now, evaluate this answersheet:
        {answersheet}
        """,
        input_variables=["answersheet"]
        )
    chain = prompt | llm | JsonOutputParser()
    return chain.invoke({"answersheet":answersheet})

def enrich_answersheet_with_context(answersheet, document_chunks, top_k=4):
    """
    For each question in the answersheet, retrieve relevant context from the document chunks
    and append it under the 'context' key.
    """
    try:
        print("DEBUG: Creating vector store from document chunks...")
        # Recreate vector store from document chunks
        vector_store = create_vector_store(document_chunks)
        print("DEBUG: Vector store created successfully")
        
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        print("DEBUG: Retriever created successfully")

        for section_name, section_data in answersheet.items():
            print(f"DEBUG: Processing section: {section_name}")
            for q in section_data['questions']:
                print(f"DEBUG: Processing question {q['question_no']}")
                # Construct query from question and optionally student answer
                query = f"{q['question']} Student's answer: {q['student_answer']}"
                
                # Retrieve relevant documents
                docs = retriever.invoke(query)
                print(f"DEBUG: Retrieved {len(docs)} documents for question {q['question_no']}")
                
                # Join document chunks as a single string
                context_text = "\n".join([doc.page_content for doc in docs])
                
                # Append to question
                q['context'] = context_text
                print(f"DEBUG: Added context of length {len(context_text)} to question {q['question_no']}")

        print("DEBUG: Enrichment process completed successfully")
        return answersheet
    except Exception as e:
        print(f"DEBUG: Error in enrich_answersheet_with_context: {str(e)}")
        traceback.print_exc()
        raise e

if __name__ == '__main__':
    app.secret_key = 'testgen-secret-key-2024'  # Required for session
    app.run(debug=True) 