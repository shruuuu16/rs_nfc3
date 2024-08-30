import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import pickle
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Load and prepare data
df = pd.read_csv('charity_navigator.csv')
df['mission'] = df['mission'].fillna('')
df['tagline'] = df['tagline'].fillna('')

le_category = LabelEncoder()
le_cause = LabelEncoder()
df['category_encoded'] = le_category.fit_transform(df['category'])
df['cause_encoded'] = le_cause.fit_transform(df['cause'])

df['combined_features'] = df['mission'] + ' ' + df['tagline'] + ' ' + df['category'] + ' ' + df['cause']

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(charity_id, cosine_sim, df):
    if charity_id not in df['charityid'].values:
        return None
    
    idx = df.index[df['charityid'] == charity_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    charity_indices = [i[0] for i in sim_scores]
    return df.iloc[charity_indices][['charityid', 'category', 'cause', 'mission', 'tagline']].to_dict(orient='records')

@app.route('/recommend', methods=['GET'])
def recommend():
    charity_id = request.args.get('charity_id')
    
    if charity_id is None:
        return jsonify({"error": "Please provide a charity ID"}), 400

    try:
        charity_id = int(charity_id)
    except ValueError:
        return jsonify({"error": "Charity ID must be an integer"}), 400

    recommendations = get_recommendations(charity_id, cosine_sim, df)
    
    if recommendations:
        return jsonify({"recommendations": recommendations})
    else:
        return jsonify({"error": "Charity ID not found"}), 404

@app.route('/')
def home():
    return "Welcome to the Charity Recommendation API. Use /recommend endpoint with a charity_id parameter."

if __name__ == '__main__':
    app.run(port=5050, debug=True)