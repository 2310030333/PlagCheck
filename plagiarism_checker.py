import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from collections import Counter
from googletrans import Translator

nltk.download("stopwords")
nltk.download("punkt")

class ArxivScraper:
    def __init__(self, max_results=5):
        self.max_results = max_results
        self.base_url = "http://export.arxiv.org/api/query"
        self.results = []

    def fetch_papers(self, query):
        """Fetches papers from arXiv API based on a dynamically generated query."""
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': self.max_results
        }
        response = requests.get(self.base_url, params=params)
        soup = BeautifulSoup(response.text, 'xml')
        entries = soup.find_all('entry')

        for entry in entries:
            title = entry.title.text
            summary = entry.summary.text
            url = entry.id.text  
            self.results.append({'title': title, 'summary': summary, 'url': url})

        return self.results

class PlagiarismChecker:
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """Cleans and tokenizes text, removing stop words."""
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        return ' '.join(words)

    def extract_keywords(self, text, num_keywords=10):
        """Extracts top keywords from text."""
        words = self.preprocess_text(text).split()
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(num_keywords)]

    def encode_text(self, text):
        """Encodes text using a pre-trained transformer model for comparison."""
        return self.model.encode([text])[0]

    def create_segments(self, text, window_size=30, overlap=15):
        """Creates overlapping segments from text for patchwork plagiarism detection."""
        words = text.split()
        segments = []
        for i in range(0, len(words) - window_size + 1, window_size - overlap):
            segment = ' '.join(words[i:i + window_size])
            segments.append(segment)
        return segments

    def check_similarity(self, user_text, scraped_papers, threshold=0.7):
        """Compares user text with scraped papers to check for patchwork plagiarism."""
        user_segments = self.create_segments(user_text)
        results = []
        total_similarity = 0
        match_count = 0
        contributions = {paper['title']: 0 for paper in scraped_papers} 

        for paper in scraped_papers:
            paper_segments = self.create_segments(paper['summary'])
            paper_results = []
            paper_total_similarity = 0  

            for user_segment in user_segments:
                user_embedding = self.encode_text(user_segment)
                for paper_segment in paper_segments:
                    paper_embedding = self.encode_text(paper_segment)
                    similarity = cosine_similarity(
                        [user_embedding], [paper_embedding]
                    )[0][0]
                    if similarity >= threshold:
                        total_similarity += similarity
                        paper_total_similarity += similarity  
                        match_count += 1
                        paper_results.append({
                            'user_segment': user_segment,
                            'paper_segment': paper_segment,
                            'similarity': similarity
                        })

            if paper_results:
                results.append({
                    'title': paper['title'],
                    'url': paper['url'],  
                    'matches': sorted(paper_results, key=lambda x: x['similarity'], reverse=True)
                })

                contributions[paper['title']] += paper_total_similarity 

        average_similarity = (total_similarity / match_count) if match_count > 0 else 0

        
        total_contribution = sum(contributions.values())
        contribution_percentages = {title: (contribution / total_contribution * 100) if total_contribution > 0 else 0
                                    for title, contribution in contributions.items()}

        return results, average_similarity, contribution_percentages

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

def process_document(pdf_path):
    
    pdf_text = extract_text_from_pdf(pdf_path)

    
    translator = Translator()
    translated_text = translator.translate(pdf_text, src='es', dest='en').text

    checker = PlagiarismChecker()
    keywords = checker.extract_keywords(translated_text)
    query = ' '.join(keywords)  

    scraper = ArxivScraper()
    papers = scraper.fetch_papers(query)

    results, average_similarity, contributions = checker.check_similarity(translated_text, papers)

    for result in results:
        print(f"\nPaper Title: {result['title']}")
        print(f"URL: {result['url']}")
        for match in result['matches']:
            print(f"Matched User Segment: {match['user_segment']}")
            print(f"Matched Paper Segment: {match['paper_segment']}")
            print(f"Similarity: {match['similarity']:.2f}")

    print(f"\nOverall Average Similarity: {average_similarity:.2f}")

    for title, contribution in contributions.items():
        print(f"Contribution of '{title}' to Overall Average Similarity: {contribution:.2f}")


