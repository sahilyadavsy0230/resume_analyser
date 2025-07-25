from flask import Flask, request, render_template
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import joblib
import nltk

nltk.download('stopwords')
nltk.download('wordnet')


label_mapping = {
    0: 'Advocate',
    1: 'Arts',
    2: 'Automation Testing',
    3: 'Blockchain',
    4: 'Business Analyst',
    5: 'Civil Engineer',
    6: 'Data Science',
    7: 'Database',
    8: 'DevOps Engineer',
    9: 'DotNet Developer',
    10: 'ETL Developer',
    11: 'Electrical Engineering',
    12: 'HR',
    13: 'Hadoop',
    14: 'Health and fitness',
    15: 'Java Developer',
    16: 'Mechanical Engineer',
    17: 'Network Security Engineer',
    18: 'Operations Manager',
    19: 'PMO',
    20: 'Python Developer',
    21: 'SAP Developer',
    22: 'Sales',
    23: 'Testing',
    24: 'Web Designing'
}


lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

def preprocess(input_text):
    cleanText = re.sub(r'http\S+\s', ' ', input_text)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)  
    cleanText = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    cleanText = re.sub(r'[^a-zA-Z]', ' ', cleanText)
    cleanText = cleanText.lower()
    tokens = cleanText.split()
    tokens = [lem.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get('resume')
    clean_text = preprocess(input_text)
    vectorized = tfidf.transform([clean_text])
    prediction = model.predict(vectorized)[0]
    prediction = label_mapping.get(prediction, "Unknown Category")
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
