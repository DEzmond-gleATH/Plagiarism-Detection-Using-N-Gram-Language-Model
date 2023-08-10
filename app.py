import os
from flask import Flask, render_template, request, url_for
from plagde import plagde
import docx
from PyPDF2 import PdfReader

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/process', methods=['POST'])
def process():
    text_input = request.form['text_input']
    file_input = request.files['file_input']
    fname = file_input.filename
    file_input.save(fname) # save the uploaded file to disk
    #print(file_input.filename)

    # Read .txt files
    if fname.endswith('.txt'):
        with open(fname, 'r') as f:
            text = f.read()
        output = plagde(text_input, text)
        os.remove(fname)
        return render_template('heatmap_output.html', output=output)

    # Read .pdf files
    elif fname.endswith('.pdf'):
        with open(fname, 'rb') as f:
            pdf_reader = PdfReader(f)
            text = ''
            for page in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[page].extract_text()

                text += page_text
        output = plagde(text_input, text)
        os.remove(fname)
        return render_template('heatmap_output.html', output=output)

    # Read .docx files
    elif fname.endswith('.docx'):
        doc = docx.Document(fname)
        text = ''
        for para in doc.paragraphs:
            text += para.text
        output = plagde(text_input, text)
        os.remove(fname)
        return render_template('heatmap_output.html', output=output)


    # Unsupported file format
    else:
        return "Invalid input"



if __name__ == '__main__':
    app.run(debug=True, port=8000)
