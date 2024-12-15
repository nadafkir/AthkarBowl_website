from django.shortcuts import render
import os
import joblib
from django.http import JsonResponse
import pandas as pd
import random
from googletrans import Translator
import unicodedata
import re
import string

def index(request):
    return render(request, 'index.html')

def athkar_bowl(request):
    return render(request, 'athkar-bowl.html')

def athkar(request):
    return render(request, 'athkar.html')

def about_us(request):
    return render(request, 'about-us.html')
      
# PATHS `.pkl`
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = os.path.join(BASE_DIR, 'sentiment_model.pkl')
# VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

##############################""

MODEL_PATH = r"C:\Users\AFKIR\OneDrive\Bureau\AthkaPerEmotion\EmotionAnalysis\app\model\emotion_model.pkl"
VECTORIZER_PATH = r"C:\Users\AFKIR\OneDrive\Bureau\AthkaPerEmotion\EmotionAnalysis\app\model\tfidf_vectorizer.pkl"
ATHKAR_PATH = r"C:\Users\AFKIR\OneDrive\Bureau\AthkaPerEmotion\EmotionAnalysis\app\datasets\adhkars_emotions.csv"

# Charger le modèle et le vectorizer
sentiment_model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
#######################################

try:
    adhkars_data = pd.read_csv(ATHKAR_PATH)
    adhkars_data.columns = adhkars_data.columns.str.strip()  
    print("Adhkars data loaded successfully.")
except Exception as e:
    print(f"Error loading adhkars data: {e}")
    adhkars_data = None

def translate_to_english(text):
    translator = Translator()
    translated = translator.translate(text, src='auto', dest='en')
    return translated.text

def unicode_to_ascii(s): 
    return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def clean_text(text):
    text = unicode_to_ascii(text.lower().strip())
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"cuz", "because", text)

    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = re.sub("(\\W)"," ",text) 
    text = re.sub('\S*\d\S*\s*','', text)
    return text

def predict_sentiment(request):
    if request.method == 'POST':
        text = request.POST.get('feelings', '')
        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)
        text_translated=translate_to_english(text)
        text_translated_cleaned=clean_text(text_translated)
        text_transformed = vectorizer.transform([text_translated_cleaned])
        prediction = sentiment_model.predict(text_transformed)

        print(f"Input text: {text}")
        print(f"Prediction: {prediction[0]}")
        print(f"translation: {text_translated_cleaned}")
        
        return text, prediction[0]  
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)

def athkar_recommendation(request):
        if adhkars_data is None:
            return JsonResponse({'error': 'Adhkars data is not available.'}, status=500)
        
        try:
            # Log pour vérifier l'appel de la fonction de prédiction
            print("Calling predict_sentiment function...")
            feelings, prediction = predict_sentiment(request)  
            print(f"Predicted emotion: {prediction}") 
            
            # Vérification des colonnes disponibles
            required_columns = ['anglais', 'arabe', 'translitteration']
            for col in required_columns:
                if col not in adhkars_data.columns:
                    return JsonResponse({'error': f'Missing column: {col}'}, status=500)
            
            # Filtrer les athkars correspondant à l'émotion prédite
            print("Filtering matching adhkars...")
            matching_adhkars = adhkars_data[adhkars_data['emotion'] == prediction][required_columns]
            
            if matching_adhkars.empty:
                return JsonResponse({
                    'text': feelings,
                    'recommended_adhkars': [],
                    'message': 'No matching Athkars found for this emotion.'
                }, status=200)
            
            # Sélectionner aléatoirement jusqu'à 3 athkars
            recommended_adhkars = matching_adhkars.sample(n=min(3, len(matching_adhkars)))
            
            # Convertir en liste de dictionnaires pour le format JSON
            recommended_adhkars_list = recommended_adhkars.to_dict(orient='records')
            print(f"Recommended Athkars: {recommended_adhkars_list}")
            
            return JsonResponse({
                'text': feelings,
                'recommended_adhkars': recommended_adhkars_list
            }, status=200)
        
        except Exception as e:
            # Log d'erreur détaillé
            print(f"Error in athkar_recommendation: {e}")
            return JsonResponse({'error': 'An error occurred.', 'details': str(e)}, status=500)
    
