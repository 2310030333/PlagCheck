from flask import Flask, render_template, request, redirect, url_for
from plagiarism_checker import ArxivScraper, PlagiarismChecker, extract_text_from_pdf, Translator
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/index')
def index():
    return render_template("index.html")
   
@app.route('/check', methods=['POST'])
def check():
    if request.method == 'POST':
        pdf_file = request.files['pdf_file']
        pdf_path = f"./uploads/{pdf_file.filename}"
        pdf_file.save(pdf_path)

        pdf_text = extract_text_from_pdf(pdf_path)

        translator = Translator()
        translated_text = translator.translate(pdf_text, src='es', dest='en').text

        checker = PlagiarismChecker()
        keywords = checker.extract_keywords(translated_text)
        query = ' '.join(keywords)  

        scraper = ArxivScraper()
        papers = scraper.fetch_papers(query)

        results, average_similarity, contributions = checker.check_similarity(translated_text, papers)

        return render_template('results.html', results=results, average_similarity=average_similarity, contributions=contributions)

if __name__ == '__main__':
    app.run(debug=True)
