import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import requests
import json
from ntscraper import Nitter
import time
import warnings


# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😁",
    layout="wide"
)
     
# Download resources and load model components
@st.cache_resource
def load_resources():
    nltk.download('stopwords')
    return stopwords.words('english'), PorterStemmer()

@st.cache_resource
def load_model_components():
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please ensure model.pkl and vectorizer.pkl exist in the current directory.")
        st.stop()

# Text preprocessing function
def preprocess_text(text, stop_words, stemmer):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

# Sentiment prediction function
def predict_sentiment(text, model, vectorizer, stop_words, stemmer):
    processed_text = preprocess_text(text, stop_words, stemmer)
    processed_text = [processed_text]
    vectorized_text = vectorizer.transform(processed_text)
    prediction = model.predict(vectorized_text)
    return "Positive" if prediction[0] == 1 else "Negative"

# Function to display sentiment with appropriate styling
def display_sentiment_result(text, sentiment):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Text:")
        st.write(text)
    
    with col2:
        if sentiment == "Positive":
            st.success("POSITIVE")
        else:
            st.error("NEGATIVE")

# Create sentiment card for multiple text items
def create_sentiment_card(text, sentiment, item_num, prefix="Text"):
    with st.container():
        st.markdown(f"### {prefix} {item_num}")
        display_sentiment_result(text, sentiment)
        st.divider()

# Function to fetch tweets with alternative methods if needed
def fetch_tweets(username, count=5, max_retries=3):
    # First try with ntscraper
    for attempt in range(max_retries):
        try:
            # Create scraper with specific instance to avoid random selection
            scraper = Nitter(instance="https://nitter.net")  # Use a specific instance instead of random
            
            try:
                # Try the newer API format first
                tweets_data = scraper.get_tweets(username, mode='user', number=count)
                if tweets_data and 'tweets' in tweets_data and tweets_data['tweets']:
                    return [tweet['text'] for tweet in tweets_data['tweets']]
            except (TypeError, AttributeError):
                try:
                    # Try the older API format as fallback
                    tweets_data = scraper.get_tweets(username, count)
                    if tweets_data and 'tweets' in tweets_data and tweets_data['tweets']:
                        return [tweet['text'] for tweet in tweets_data['tweets']]
                except Exception as e:
                    st.warning(f"Error with older API format: {str(e)}")
                    
            # Try alternative instances if the first one fails
            alternative_instances = [
                "https://nitter.cz", 
                "https://nitter.it",
                "https://nitter.poast.org"
            ]
            
            for instance in alternative_instances:
                try:
                    alt_scraper = Nitter(instance=instance)
                    tweets_data = alt_scraper.get_tweets(username, mode='user', number=count)
                    if tweets_data and 'tweets' in tweets_data and tweets_data['tweets']:
                        return [tweet['text'] for tweet in tweets_data['tweets']]
                except Exception:
                    continue  # Try next instance
                    
        except Exception as e:
            time.sleep(1)  # Wait between retries
    
    # Fallback option: Generate mock tweets for demo purposes
    st.warning("⚠️ Unable to fetch real tweets. Using sample tweets for demonstration.")
    return [
        f"I'm really enjoying using the products from {username}! #happy #customer",
        f"Just had a terrible experience with {username}'s customer service. Very disappointed.",
        f"The new release from {username} has some great features but also some bugs.",
        f"Can't believe how amazing {username}'s new product is! Absolutely worth every penny!",
        f"{username} needs to improve their response time. Been waiting for days for a reply."
    ][:count]

# Main app function
def main():
    # Load resources
    stop_words, stemmer = load_resources()
    model, vectorizer = load_model_components()
    
    # App title and description
    st.title("Sentiment Analysis Dashboard")
    st.markdown("""
    This application uses machine learning to analyze the sentiment of text as either Positive or Negative.
    """)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["Single Text Analysis", "Batch Analysis", "Twitter Analysis", "About"])
    
    # Tab 1: Single Text Analysis
    with tab1:
        st.header("Analyze Text Sentiment")
        text_input = st.text_area("Enter text to analyze:", height=150)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Analyze", type="primary"):
                if text_input:
                    with st.spinner("Analyzing sentiment..."):
                        sentiment = predict_sentiment(text_input, model, vectorizer, stop_words, stemmer)
                        st.session_state.result = {"text": text_input, "sentiment": sentiment}
                else:
                    st.warning("Please enter some text to analyze.")
        
        # Display result if available
        if "result" in st.session_state:
            st.subheader("Analysis Result")
            display_sentiment_result(st.session_state.result["text"], st.session_state.result["sentiment"])
    
    # Tab 2: Batch Analysis
    with tab2:
        st.header("Batch Text Analysis")
        st.markdown("Analyze multiple text entries at once. Enter one text per line.")
        
        batch_text = st.text_area("Enter multiple texts (one per line):", height=200)
        
        if st.button("Analyze Batch", type="primary"):
            if batch_text:
                texts = [text.strip() for text in batch_text.split('\n') if text.strip()]
                
                if texts:
                    with st.spinner(f"Analyzing {len(texts)} text entries..."):
                        st.session_state.batch_results = []
                        
                        for text in texts:
                            sentiment = predict_sentiment(text, model, vectorizer, stop_words, stemmer)
                            st.session_state.batch_results.append({
                                "text": text,
                                "sentiment": sentiment
                            })
                else:
                    st.warning("No valid text entries found.")
            else:
                st.warning("Please enter some text to analyze.")
        
        # File uploader for CSV
        st.subheader("Or upload a CSV file")
        st.write("The CSV should have a column named 'text' containing the texts to analyze.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    if st.button("Analyze CSV Data", type="primary"):
                        with st.spinner(f"Analyzing {len(df)} text entries from CSV..."):
                            st.session_state.batch_results = []
                            
                            for _, row in df.iterrows():
                                text = str(row['text'])
                                sentiment = predict_sentiment(text, model, vectorizer, stop_words, stemmer)
                                st.session_state.batch_results.append({
                                    "text": text,
                                    "sentiment": sentiment
                                })
                else:
                    st.error("CSV file must contain a column named 'text'.")
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")
        
        # Display batch results if available
        if "batch_results" in st.session_state and st.session_state.batch_results:
            # Display batch results
            display_batch_results(st.session_state.batch_results)
    
    # Tab 3: Twitter Analysis
    with tab3:
        st.header("Twitter Sentiment Analysis")
        st.markdown("""
        Fetch and analyze tweets from a Twitter user. 
        **Note**: This feature uses web scraping which may sometimes be unreliable.
        """)
        
        username = st.text_input("Enter Twitter username (without @):")
        tweet_count = st.slider("Number of tweets to analyze:", min_value=1, max_value=10, value=5)
        
        # Add option to use sample tweets directly
        use_sample = st.checkbox("Use sample tweets (skip fetching)", value=False)
        
        if st.button("Fetch & Analyze Tweets", type="primary"):
            if username:
                with st.spinner(f"Fetching tweets from @{username}..."):
                    try:
                        if use_sample:
                            # Use sample tweets directly if checkbox is selected
                            tweets = [
                                f"I'm really enjoying using the products from {username}! #happy #customer",
                                f"Just had a terrible experience with {username}'s customer service. Very disappointed.",
                                f"The new release from {username} has some great features but also some bugs.",
                                f"Can't believe how amazing {username}'s new product is! Absolutely worth every penny!",
                                f"{username} needs to improve their response time. Been waiting for days for a reply."
                            ][:tweet_count]
                            st.info("Using sample tweets as requested.")
                        else:
                            # Try to fetch real tweets
                            tweets = fetch_tweets(username, tweet_count)
                        
                        if tweets:
                            st.session_state.twitter_results = []
                            
                            for tweet_text in tweets:
                                sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words, stemmer)
                                st.session_state.twitter_results.append({
                                    "text": tweet_text,
                                    "sentiment": sentiment
                                })
                        else:
                            st.error("No tweets found.")
                    except Exception as e:
                        st.error(f"Error analyzing tweets: {str(e)}")
            else:
                st.warning("Please enter a Twitter username.")
        
        # Display Twitter results if available
        if "twitter_results" in st.session_state and st.session_state.twitter_results:
            st.subheader("Twitter Analysis Results")
            
            # Statistics
            pos_count = sum(1 for result in st.session_state.twitter_results if result["sentiment"] == "Positive")
            neg_count = len(st.session_state.twitter_results) - pos_count
            
            # Create metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Tweets", len(st.session_state.twitter_results))
            with col2:
                st.metric("Positive Tweets", pos_count)
            with col3:
                st.metric("Negative Tweets", neg_count)
            
            # Create sentiment distribution chart
            chart_data = pd.DataFrame({
                'Sentiment': ['Positive', 'Negative'],
                'Count': [pos_count, neg_count]
            })
            
            st.subheader("Sentiment Distribution")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.bar_chart(chart_data.set_index('Sentiment'))
            with col2:
                # Calculate percentage
                total = pos_count + neg_count
                if total > 0:
                    st.write(f"Positive: {pos_count/total*100:.1f}%")
                    st.write(f"Negative: {neg_count/total*100:.1f}%")
                    
                    # Simple gauge
                    sentiment_score = pos_count/total
                    st.progress(sentiment_score)
                    
                    if sentiment_score > 0.7:
                        st.success("Very Positive Sentiment")
                    elif sentiment_score > 0.5:
                        st.info("Somewhat Positive Sentiment")
                    elif sentiment_score > 0.3:
                        st.warning("Somewhat Negative Sentiment")
                    else:
                        st.error("Very Negative Sentiment")
            
            # Download results as CSV
            twitter_df = pd.DataFrame(st.session_state.twitter_results)
            csv = twitter_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download Twitter Analysis as CSV",
                data=csv,
                file_name=f"{username}_twitter_sentiment.csv",
                mime="text/csv",
            )
            
            # Display individual tweet results with filtering
            st.subheader("Individual Tweet Analysis")
            
            # Add a filter
            filter_option = st.selectbox("Filter tweets by sentiment:", ["All", "Positive Only", "Negative Only"])
            
            # Apply filter
            filtered_results = st.session_state.twitter_results
            if filter_option == "Positive Only":
                filtered_results = [r for r in filtered_results if r["sentiment"] == "Positive"]
            elif filter_option == "Negative Only":
                filtered_results = [r for r in filtered_results if r["sentiment"] == "Negative"]
            
            for i, result in enumerate(filtered_results):
                create_sentiment_card(result["text"], result["sentiment"], i+1, prefix="Tweet")
    
    # Tab 4: About
    with tab4:
        st.header("About This App")
        st.markdown("""
        ### Sentiment Analysis Application
        
        This application uses a machine learning model trained on a large dataset of tweets to predict whether text has a positive or negative sentiment.
        
        #### How it works:
        1. Text is preprocessed by removing special characters, converting to lowercase, and stemming words
        2. The processed text is then transformed into numerical features using TF-IDF vectorization
        3. A logistic regression model predicts whether the sentiment is positive or negative
        
        #### Features:
        - Single text analysis for quick sentiment checks
        - Batch analysis for processing multiple texts at once
        - Twitter analysis to fetch and analyze tweets from specific users
        - CSV file upload for analyzing large datasets
        - Download analysis results as CSV
        - Filter and view results by sentiment
        
        #### Technologies used:
        - Streamlit for the web application
        - NLTK for natural language processing
        - scikit-learn for machine learning components
        - ntscraper for Twitter data retrieval
        - pandas for data handling
        
        #### Model Information:
        The model was trained on a dataset of 1.6 million tweets, with preprocessing steps matching those used in this application.
        """)

# Helper function to display batch results
def display_batch_results(results):
    st.subheader("Batch Analysis Results")
    
    # Statistics
    pos_count = sum(1 for result in results if result["sentiment"] == "Positive")
    neg_count = len(results) - pos_count
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Texts", len(results))
    with col2:
        st.metric("Positive", pos_count)
    with col3:
        st.metric("Negative", neg_count)
    
    # Create chart
    chart_data = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative'],
        'Count': [pos_count, neg_count]
    })
    
    st.subheader("Sentiment Distribution")
    st.bar_chart(chart_data.set_index('Sentiment'))
    
    # Download results as CSV
    results_df = pd.DataFrame(results)
    csv = results_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="sentiment_analysis_results.csv",
        mime="text/csv",
    )
    
    # Display individual text results
    st.subheader("Individual Text Analysis")
    
    # Add a filter
    filter_option = st.selectbox("Filter results by sentiment:", ["All", "Positive Only", "Negative Only"])
    
    # Apply filter
    filtered_results = results
    if filter_option == "Positive Only":
        filtered_results = [r for r in filtered_results if r["sentiment"] == "Positive"]
    elif filter_option == "Negative Only":
        filtered_results = [r for r in filtered_results if r["sentiment"] == "Negative"]
    
    for i, result in enumerate(filtered_results):
        create_sentiment_card(result["text"], result["sentiment"], i+1)

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


if __name__ == "__main__":
    main()
