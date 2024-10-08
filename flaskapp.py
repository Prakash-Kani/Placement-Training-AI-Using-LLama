
from flask import Flask, request, jsonify
from doc_loader import ingest
from Question_Generator import *
from Evaluation import *
from Mock_Interview import *
from datetime import datetime as dt
import os

app = Flask(__name__)


UPLOAD_FOLDER = 'Uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


DB_FOLDER = 'Databases'
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)
app.config['DB_FOLDER'] = DB_FOLDER


@app.route('/Document-Upload', methods=['POST'])
def ingest_pdf():
    # Ensure that 'filename' and 'pdf' are part of the form data
    if 'filename' not in request.form :
        return jsonify({'error': 'Filename is missing'}), 400
    if  'pdf' not in request.files:
        return jsonify({'error': ' PDF file is missing'}), 400
    # Get the filename and PDF file from the request
    filename = request.form['filename']
    pdf_file = request.files['pdf']
    # topic = request.form['topic']
    # history = f"Let's dive into {topic}. What specific topic or question can I help you with today?"

    # Check if the file is a valid PDF
    if pdf_file.filename == '' or not pdf_file.filename.endswith('.pdf'):
        return jsonify({'error': 'Invalid file format. Please upload a PDF.'}), 400

    # Save the PDF file to the server
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf_file.save(file_path)

    # if topic:
    #     text_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{filename}.txt')
    #     with open(text_file_path, 'w') as text_file:
    #         text_file.write(history)


    persist_directory = os.path.join(app.config['DB_FOLDER'], filename)
    # print(file_path, persist_directory)
    ingest(file_path=file_path, persist_directory = persist_directory)


    # Return success response
    return jsonify({'message': 'File uploaded and processed successfully'}), 200



@app.route('/Question-Generator', methods=['POST'])
def question_generator():
    # Get the input data from the request
    data = request.get_json()

    # Ensure the input is provided
    if 'course_name' not in data:
        return jsonify({'error': 'No course_name provided'}), 400
    if 'question_level' not in data:
        return jsonify({'error': 'No question_level provided'}), 400
    if 'question_type' not in data:
        return jsonify({'error': 'No question_type provided'}), 400
    if 'session_id' not in data:
        return jsonify({'error': 'No session_id provided'}), 400

    filename = data['course_name']
    question_level = data['question_level']
    question_type = data['question_type']
    session_id = data['session_id']
    topic_name = None
    persist_directory = os.path.join(app.config['DB_FOLDER'], filename)

    if 'topic_name' in data:
        topic_name = data['topic_name']
    


    if filename and question_level and question_type and session_id and topic_name:
        prompt = f'Generate a {question_level} level {question_type} question from the {topic_name}'
    elif filename and question_level and question_type and session_id:
        prompt = f'Generate a {question_level} level {question_type} question'

    if prompt and filename and session_id:
        print(prompt)
        question_generation_chain = Question_Generator_Chain(persist_directory)
        result= question_generation_chain.invoke({"input": prompt},
                                                    config={"configurable": {"session_id": session_id}})["answer"]

        response = {'question': result, 'time_stamp': dt.now()}
        # Return the response as JSON
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid Course Name'}), 400



@app.route('/Evaluation', methods=['POST'])
def get_evaluation():
    # Get the input data from the request
    data = request.get_json()

    # Ensure the input is provided
    if 'course_name' not in data:
        return jsonify({'error': 'No course_name provided'}), 400
    if 'question_level' not in data:
        return jsonify({'error': 'No question_level provided'}), 400
    if 'question_type' not in data:
        return jsonify({'error': 'No question_type provided'}), 400
    if 'session_id' not in data:
        return jsonify({'error': 'No session_id provided'}), 400
    if 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    if 'answer' not in data:
        return jsonify({'error': 'No answer provided'}), 400

    filename = data['course_name']
    question_level = data['question_level']
    question_type = data['question_type']
    session_id = data['session_id']
    question = data['question']
    answer = data['answer']
    persist_directory = os.path.join(app.config['DB_FOLDER'], filename)


    


    if filename and question_level and question_type and session_id and question and answer:
        prompt = f'''Evaluate  the  {question_level} level {question_type} 
                    Question:\n
                    {question}
                    Answer:\n
                    {answer}'''
    # elif filename and question_level and question_type and session_id:
    #     prompt = f'Generate a {question_level} level {question_type} question'

    if prompt and filename and session_id:
        # print(prompt)
        evaluation_chain = Evaluation_Conversational_Chain(persist_directory)
        result= evaluation_chain.invoke({"input": prompt},
                                                    config={"configurable": {"session_id": session_id}})["answer"]
    
        # result = chain.invoke("generate the report")

        # response = {'report': result, 'time_stamp': dt.now()}
        # print(type(result), result)
        response = {'evaluation': result, 'time_stamp': dt.now()}
        # Return the response as JSON
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid Course Name'}), 400
    


@app.route('/Mock-Interview', methods=['POST'])
def get_interview():
    # Get the input data from the request
    data = request.get_json()

    # Ensure the input is provided
    
    if 'session_id' not in data:
        return jsonify({'error': 'No session_id provided'}), 400
    if 'position' not in data:
        return jsonify({'error': 'No position provided'}), 400
    if 'role' not in data:
        return jsonify({'error': 'No role provided'}), 400
    if 'skills' not in data:
        return jsonify({'error': 'No skills provided'}), 400
    if 'answer' not in data:
        return jsonify({'error': 'No answer provided'}), 400

    session_id = data['session_id']
    position = data['position']
    role = data['role']
    skills = data['skills']
    answer = data['answer']
    persist_directory = 'Databases\new' #os.path.join(app.config['DB_FOLDER'], filename)
    if answer =="" or answer == None or answer ==" ":
        prompt = f"start the placement training for {position} {role} role required skills are {skills}"
    else:
        prompt = answer
    if prompt  and session_id:
        evaluation_chain = Mock_Interview_Chain(persist_directory)
        result= evaluation_chain.invoke({"input": prompt},
                                                    config={"configurable": {"session_id": session_id}})["answer"]
    
        
        response = {'question': result, "session_id": session_id, 'time_stamp': dt.now()}
        # Return the response as JSON
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid Course Name'}), 400
    


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug=True)
