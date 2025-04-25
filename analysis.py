import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import chi2_contingency, spearmanr, mannwhitneyu
import re
import warnings
warnings.filterwarnings('ignore')

class NetflixAnalyzer:
    def _init_(self, file_path):
        """Initialize with the path to the Netflix reviews CSV file."""
        self.df = pd.read_csv('/Users/AnkitaMac/Desktop/cleaned_netflix_reviews.csv')
        self.processed_df = None
        self.model = None
        print(f"Loaded dataset with shape: {self.df.shape}")

    def preprocess_data(self):
        """Clean and preprocess the data."""
        print("\n=== Data Preprocessing ===")
        
        # Display initial info
        print("\nInitial missing values:")
        print(self.df.isnull().sum())

        # Basic cleaning
        self.processed_df = self.df.copy()
        self.processed_df.drop_duplicates(subset='reviewId', inplace=True)
        self.processed_df.dropna(subset=['content'], inplace=True)
        
        # Clean text content
        self.processed_df['cleaned_content'] = self.processed_df['content'].apply(self._clean_text)
        
        # Add basic features
        self.processed_df['review_length'] = self.processed_df['cleaned_content'].str.len()
        self.processed_df['word_count'] = self.processed_df['cleaned_content'].apply(lambda x: len(str(x).split()))
        
        print(f"\nShape after cleaning: {self.processed_df.shape}")
        return self.processed_df

    def perform_sentiment_analysis(self):
        """Conduct sentiment analysis on reviews."""
        print("\n=== Sentiment Analysis ===")
        
        # Calculate sentiment scores
        self.processed_df['sentiment_polarity'] = self.processed_df['cleaned_content'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        self.processed_df['sentiment_subjectivity'] = self.processed_df['cleaned_content'].apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity
        )
        
        # Categorize sentiments
        self.processed_df['sentiment_category'] = pd.cut(
            self.processed_df['sentiment_polarity'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['Negative', 'Neutral', 'Positive']
        )
        
        # Calculate satisfaction score
        self.processed_df['satisfaction_score'] = (
            (self.processed_df['sentiment_polarity'] + 1) * 25 +
            self.processed_df['score'] * 6 +
            np.log1p(self.processed_df['thumbsUpCount']) * 5
        ).clip(0, 100)
        
        # Display sentiment distribution
        sentiment_stats = self.processed_df['sentiment_category'].value_counts()
        print("\nSentiment Distribution:")
        print(sentiment_stats)
        
        # Plot sentiment distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.processed_df, x='sentiment_polarity', bins=50)
        plt.title('Distribution of Sentiment Polarity')
        plt.show()

    def perform_clustering(self):
        """Perform customer segmentation using clustering."""
        print("\n=== Clustering Analysis ===")
        
        # Prepare features for clustering
        features = ['satisfaction_score', 'sentiment_polarity', 'score', 
                   'thumbsUpCount', 'review_length', 'word_count']
        X = self.processed_df[features].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.processed_df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_stats = self.processed_df.groupby('cluster')[features].mean()
        print("\nCluster Centers:")
        print(cluster_stats)
        
        # Visualize clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.processed_df,
            x='sentiment_polarity',
            y='satisfaction_score',
            hue='cluster',
            palette='deep'
        )
        plt.title('Clusters based on Sentiment and Satisfaction')
        plt.show()

    def perform_hypothesis_tests(self):
        """Conduct statistical hypothesis tests."""
        print("\n=== Hypothesis Testing ===")
        
        # Test 1: Correlation between sentiment and rating
        correlation, p_value = spearmanr(
            self.processed_df['sentiment_polarity'],
            self.processed_df['score']
        )
        print("\nH1: Correlation between sentiment and rating")
        print(f"Correlation: {correlation:.4f}")
        print(f"P-value: {p_value:.4e}")
        
        # Test 2: Relationship between sentiment and review length
        pos_lengths = self.processed_df[self.processed_df['sentiment_category'] == 'Positive']['review_length']
        neg_lengths = self.processed_df[self.processed_df['sentiment_category'] == 'Negative']['review_length']
        stat, p_value = mannwhitneyu(pos_lengths, neg_lengths)
        print("\nH2: Difference in review length between positive and negative reviews")
        print(f"Statistic: {stat:.4f}")
        print(f"P-value: {p_value:.4e}")

    def train_predictive_model(self):
        """Train a predictive model for sentiment classification."""
        print("\n=== Predictive Modeling ===")
        
        # Prepare features and target
        features = ['score', 'review_length', 'word_count', 'sentiment_polarity']
        X = self.processed_df[features]
        y = (self.processed_df['sentiment_category'] == 'Positive').astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=30, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance_df)

    @staticmethod
    def _clean_text(text):
        """Clean text data."""
        if isinstance(text, str):
            text = re.sub(r'[^\w\s]', ' ', text)
            text = ' '.join(text.split())
            return text.lower()
        return ''

def main():
    # Initialize analyzer with your dataset
    analyzer = NetflixAnalyzer("C:/Users/Hrithik/30 Days of Python/cleaned_netflix_reviews.csv") 
    
    # Run analysis pipeline
    analyzer.preprocess_data()
    analyzer.perform_sentiment_analysis()
    analyzer.perform_clustering()
    analyzer.perform_hypothesis_tests()
    analyzer.train_predictive_model()

if _name_ == "_main_":
    main()

