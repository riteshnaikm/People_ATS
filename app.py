from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import google.generativeai as genai
import os
import PyPDF2 as pdf
from docx import Document
from dotenv import load_dotenv
import json
import re
import sqlite3
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure file upload
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

# Database initialization
def init_db():
    conn = sqlite3.connect('resume_evaluator.db')
    cursor = conn.cursor()
    # Evaluations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS evaluations (
        id TEXT PRIMARY KEY,
        filename TEXT,
        job_title TEXT,
        rank_score INTEGER,
        missing_keywords TEXT,
        profile_summary TEXT,
        evaluated_at TIMESTAMP
    )
    ''')
    # Feedback table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluation_id TEXT,
        rating INTEGER,
        comments TEXT,
        submitted_at TIMESTAMP,
        FOREIGN KEY (evaluation_id) REFERENCES evaluations (id)
    )
    ''')
    # Interview questions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS interview_questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluation_id TEXT,
        technical_questions TEXT,
        nontechnical_questions TEXT,
        generated_at TIMESTAMP,
        FOREIGN KEY (evaluation_id) REFERENCES evaluations (id)
    )
    ''')
    conn.commit()
    conn.close()

init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_gemini_response(input_prompt):
    model = genai.GenerativeModel('gemini-2.0-flash')
    try:
        response = model.generate_content(input_prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return None, str(e)


def extract_text_from_file(file_path):
    try:
        ext = file_path.rsplit('.', 1)[1].lower()
        if ext == 'pdf':
            try:
                reader = pdf.PdfReader(file_path)
                if reader.is_encrypted:
                    # Attempt to decrypt with an empty password
                    try:
                        reader.decrypt('')
                    except Exception as e:
                        return None, "PDF is encrypted and couldn't be decrypted. Error: " + str(e)
                text = ""
                for page in range(len(reader.pages)):
                    text += reader.pages[page].extract_text() or ""
                return text
            except ModuleNotFoundError as e:
                if "PyCryptodome" in str(e) or "Crypto" in str(e):
                    return None, "PyCryptodome is required for encrypted PDFs. Please install it with 'pip install pycryptodome'."
                raise
        elif ext == 'docx':
            doc = Document(file_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        elif ext == 'doc':
            return None, "Support for .doc files is limited. Please convert to .docx or PDF."
        else:
            return None, "Unsupported file format."
    except Exception as e:
        logger.error(f"File extraction error: {str(e)}")
        return None, str(e)


def save_evaluation(eval_id, filename, job_title, rank_score, missing_keywords, profile_summary):
    try:
        conn = sqlite3.connect('resume_evaluator.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO evaluations (id, filename, job_title, rank_score, missing_keywords, profile_summary, evaluated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (eval_id, filename, job_title, rank_score, json.dumps(missing_keywords), profile_summary, datetime.now())
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Database error in save_evaluation: {str(e)}")
        return False

def save_feedback(evaluation_id, rating, comments):
    try:
        conn = sqlite3.connect('resume_evaluator.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback (evaluation_id, rating, comments, submitted_at) VALUES (?, ?, ?, ?)",
            (evaluation_id, rating, comments, datetime.now())
        )
        conn.commit()
        logger.debug(f"Feedback inserted: evaluation_id={evaluation_id}, rating={rating}, comments={comments}")
        conn.close()
        return True
    except sqlite3.Error as e:
        logger.error(f"Database error in save_feedback: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in save_feedback: {str(e)}")
        return False

def save_interview_questions(evaluation_id, technical_questions, nontechnical_questions):
    try:
        conn = sqlite3.connect('resume_evaluator.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO interview_questions (evaluation_id, technical_questions, nontechnical_questions, generated_at) VALUES (?, ?, ?, ?)",
            (evaluation_id, json.dumps(technical_questions), json.dumps(nontechnical_questions), datetime.now())
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Database error in save_interview_questions: {str(e)}")
        return False

# Prompt templates
input_prompt_template = """
Act as a highly skilled ATS (Applicant Tracking System) specializing in evaluating resumes for jobs description provided. Your information will be consumed by fellow HR person so that it will help
him/her to evaluate the resumes quicker

### Task:
Evaluate the provided **resume** against the given **job description**.  Consider industry trends and the competitive job market.

### Output:
Return a valid JSON object ONLY.  The JSON object MUST have the following keys:

*   `"JD Match"` (string): Percentage match (e.g., "85%").
*   `"MissingKeywords"` (list): List of missing keywords (can be empty).
*   `"Profile Summary"` (string): Summary of strengths and areas for improvement.
*   `"Extra Info"` (string): Anything extra that can help the HR to make decision. 

Do NOT include any additional text, explanations, or formatting outside the JSON object.

---
**Resume:** {resume_text}
**Job Description:** {job_description}
"""

interview_questions_prompt = """
You are an experienced technical recruiter preparing for an interview. Based on the candidate's resume and the job description, generate relevant interview questions.

### Task:
Generate two sets of interview questions - 10 technical questions and 10 non-technical questions that are specifically tailored to assess this candidate for this role.

### Output:
Return a valid JSON object ONLY with the following keys:
* `"TechnicalQuestions"` (array): 10 technical questions related to the candidate's skills and the job requirements.
* `"NonTechnicalQuestions"` (array): 10 behavioral, situational, or cultural fit questions.

Each question should be thoughtful, specific to the resume and job description, and reveal important information about the candidate's suitability.

Do NOT include any additional text, explanations, or formatting outside the JSON object.

---
**Resume:** {resume_text}
**Job Description:** {job_description}
**Candidate Profile Summary:** {profile_summary}
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file provided'}), 400
    
    file = request.files['resume']
    job_description = request.form.get('job_description', '')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not job_description:
        return jsonify({'error': 'Job description is required'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        resume_text_result = extract_text_from_file(file_path)
        if isinstance(resume_text_result, tuple):
            return jsonify({'error': f'Error reading file: {resume_text_result[1]}'}), 500
        
        resume_text = resume_text_result
        formatted_prompt = input_prompt_template.format(resume_text=resume_text, job_description=job_description)
        
        response_text = get_gemini_response(formatted_prompt)
        if isinstance(response_text, tuple):
            return jsonify({'error': f'Error with Gemini API: {response_text[1]}'}), 500
        
        try:
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not match:
                return jsonify({'error': 'Invalid response: Could not find JSON in Gemini\'s output'}), 500
            
            json_string = match.group(0)
            response_dict = json.loads(json_string)
            
            match_percentage_str = response_dict.get("JD Match", "0%")
            match_percentage = int(match_percentage_str.strip('%'))
            missing_keywords = response_dict.get("MissingKeywords", [])
            profile_summary = response_dict.get("Profile Summary", "No summary provided.")
            extra_info = response_dict.get("Extra Info", "")
            
            eval_id = str(uuid.uuid4())
            job_title = request.form.get('job_title', 'Not specified')
            db_success = save_evaluation(
                eval_id, filename, job_title, match_percentage, missing_keywords, profile_summary
            )
            
            if not db_success:
                logger.warning("Failed to save evaluation to database")
                flash("Warning: Failed to save evaluation to database", "warning")
            
            # Generate interview questions
            questions_prompt = interview_questions_prompt.format(
                resume_text=resume_text, 
                job_description=job_description,
                profile_summary=profile_summary
            )
            
            questions_response = get_gemini_response(questions_prompt)
            
            technical_questions = []
            nontechnical_questions = []
            
            if not isinstance(questions_response, tuple):
                try:
                    match = re.search(r"\{.*\}", questions_response, re.DOTALL)
                    if match:
                        questions_json = json.loads(match.group(0))
                        technical_questions = questions_json.get("TechnicalQuestions", [])
                        nontechnical_questions = questions_json.get("NonTechnicalQuestions", [])
                        
                        # Save questions to database
                        save_interview_questions(eval_id, technical_questions, nontechnical_questions)
                except Exception as e:
                    logger.error(f"Error processing interview questions: {str(e)}")
            
            return jsonify({
                'id': eval_id,
                'match_percentage': match_percentage,
                'match_percentage_str': match_percentage_str,
                'missing_keywords': missing_keywords,
                'profile_summary': profile_summary,
                'extra_info': extra_info,
                'technical_questions': technical_questions,
                'nontechnical_questions': nontechnical_questions
            })
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error processing response: {str(e)}")
            return jsonify({'error': f'Error processing response: Invalid JSON format. {e}'}), 500
    
    return jsonify({'error': 'Invalid file format. Please upload a PDF, DOC, or DOCX file.'}), 400

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        # Check if the request is JSON
        if request.is_json:
            data = request.get_json()
            logger.debug(f"Received JSON feedback data: {data}")
        else:
            # Handle form data
            data = {
                'evaluation_id': request.form.get('evaluation_id'),
                'rating': request.form.get('rating'),
                'comments': request.form.get('comments', '')
            }
            logger.debug(f"Received form feedback data: {data}")
        
        if not data:
            logger.error("No data received in feedback request")
            return jsonify({'error': 'No data provided'}), 400
        
        evaluation_id = data.get('evaluation_id')
        rating = data.get('rating')
        comments = data.get('comments', '')
        
        # Convert rating to integer if it's a string
        if isinstance(rating, str) and rating.isdigit():
            rating = int(rating)

        logger.debug(f"Processing feedback: evaluation_id={evaluation_id}, rating={rating}, comments={comments}")

        if not evaluation_id or rating is None:
            logger.warning("Missing evaluation_id or rating")
            return jsonify({'error': 'Evaluation ID and rating are required'}), 400
        
        if not isinstance(rating, int) or rating < 1 or rating > 5:
            logger.warning(f"Invalid rating value: {rating}")
            return jsonify({'error': 'Rating must be an integer between 1 and 5'}), 400

        if save_feedback(evaluation_id, rating, comments):
            logger.info(f"Feedback saved successfully for evaluation_id={evaluation_id}")
            return jsonify({'message': 'Feedback saved successfully'}), 200
        else:
            logger.error("Failed to save feedback to database")
            return jsonify({'error': 'Failed to save feedback'}), 500
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/history')
def history():
    conn = sqlite3.connect('resume_evaluator.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluations ORDER BY evaluated_at DESC")
    rows = cursor.fetchall()
    # Convert evaluated_at strings to datetime objects and missing_keywords from JSON
    evaluations = []
    for row in rows:
        row_dict = dict(row)
        try:
            row_dict['evaluated_at'] = datetime.strptime(row['evaluated_at'], '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            row_dict['evaluated_at'] = datetime.strptime(row['evaluated_at'], '%Y-%m-%d %H:%M:%S')
        
        # Convert missing_keywords from JSON string to list
        try:
            row_dict['missing_keywords'] = json.loads(row['missing_keywords'])
        except (json.JSONDecodeError, TypeError):
            row_dict['missing_keywords'] = []
            
        evaluations.append(row_dict)
    conn.close()
    return render_template('history.html', evaluations=evaluations)

@app.route('/feedback_history')
def feedback_history():
    conn = sqlite3.connect('resume_evaluator.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT f.*, e.filename, e.job_title 
        FROM feedback f 
        JOIN evaluations e ON f.evaluation_id = e.id 
        ORDER BY f.submitted_at DESC
    """)
    feedback_entries = cursor.fetchall()
    conn.close()
    return render_template('feedback_history.html', feedback_entries=feedback_entries)

@app.route('/get_interview_questions/<evaluation_id>', methods=['GET'])
def get_interview_questions(evaluation_id):
    try:
        conn = sqlite3.connect('resume_evaluator.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT technical_questions, nontechnical_questions FROM interview_questions WHERE evaluation_id = ?", 
            (evaluation_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            technical_questions = json.loads(result['technical_questions'])
            nontechnical_questions = json.loads(result['nontechnical_questions'])
            return jsonify({
                'technical_questions': technical_questions,
                'nontechnical_questions': nontechnical_questions
            })
        else:
            return jsonify({
                'technical_questions': [],
                'nontechnical_questions': []
            })
    except Exception as e:
        logger.error(f"Error retrieving interview questions: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)