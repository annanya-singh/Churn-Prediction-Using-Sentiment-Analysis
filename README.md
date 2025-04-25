# Churn-Prediction-Using-Sentiment-Analysis
**Project Overview**

This project, submitted as part of coursework for the Department of Statistics at Pondicherry University, focuses on enhancing customer churn prediction for Netflix by integrating sentiment analysis and hypothesis testing. By analyzing user reviews from the Google Play Store, we develop a predictive model that leverages sentiment-derived Satisfaction Scores to identify at-risk users and provide actionable insights for improving customer retention.

**Objectives**





Sentiment Analysis: Quantify user sentiment from Netflix reviews to create a Satisfaction Score, capturing emotional and experiential factors influencing churn.



Churn Risk Prediction: Use clustering and predictive modeling to segment users by churn risk based on satisfaction and engagement metrics.



Hypothesis Testing: Validate relationships between sentiment, ratings, and review length to ensure the robustness of the predictive model.



Actionable Insights: Provide Netflix with data-driven recommendations to enhance user satisfaction and reduce churn through targeted retention strategies.

**Methodology**

The project follows a structured pipeline:





Data Collection: Web scraping of Netflix app reviews from the Google Play Store using a Python script, resulting in a dataset of over 98,000 reviews with attributes like review text, ratings, and engagement metrics.



Data Preprocessing: Cleaning text data, handling missing values, and calculating features such as review length and word count.



Sentiment Analysis: Using TextBlob to compute sentiment polarity and categorize reviews as Positive, Neutral, or Negative, followed by generating a Satisfaction Score.



Clustering: Applying KMeans clustering to segment users into three groups based on satisfaction and sentiment, identifying high-risk churn clusters.



Hypothesis Testing: Conducting Spearman’s correlation and Mann-Whitney U tests to explore relationships between sentiment, ratings, and review length.



Predictive Modeling: Training a Random Forest classifier to predict positive sentiment with high accuracy, using features like sentiment polarity and ratings.



Visualization: Generating histograms, scatter plots, and box plots to illustrate sentiment distribution, clustering results, and churn risk patterns.

**Key Findings**





Sentiment Distribution: Most reviews are positive (50,685), followed by neutral (30,401) and negative (16,712), indicating general user satisfaction but highlighting areas for improvement.



Clustering Insights:





Cluster 0: Low satisfaction, negative sentiment, and high churn risk.



Cluster 1: High satisfaction, positive sentiment, and low churn risk.



Cluster 2: Moderate satisfaction, neutral sentiment, with detailed reviews suggesting mixed experiences.



Hypothesis Testing:





A strong positive correlation (0.5825) exists between sentiment polarity and ratings.



Negative reviews are significantly longer than positive ones, indicating detailed feedback from dissatisfied users.



Predictive Model: The Random Forest classifier achieved 100% accuracy in identifying positive reviews, with sentiment polarity as the most influential feature.

**Recommendations**





Implement real-time sentiment monitoring to detect dissatisfaction early.



Deploy targeted retention strategies for high-risk users and engage neutral reviewers for feedback.



Optimize content and messaging based on sentiment trends to enhance user satisfaction.



Analyze negative reviews in-depth to address recurring pain points.

**Repository Structure**





netflix_reviews.csv: Dataset containing cleaned Netflix review data.



web_scraping.py: Python script for scraping reviews from the Google Play Store.



analysis.py: Python script for data preprocessing, sentiment analysis, clustering, hypothesis testing, and predictive modeling.



media/: Directory containing visualization outputs (e.g., histograms, scatter plots).



README.md: Project description and setup instructions.

**Setup Instructions**





Clone the Repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name



Install Dependencies: Ensure Python 3.8+ is installed, then install required libraries:

pip install pandas numpy matplotlib seaborn textblob scikit-learn scipy google-play-scraper



Run the Analysis:





Update the file path in analysis.py to point to netflix_reviews.csv.



Execute the analysis script:

python analysis.py



Scrape New Data (optional):





Run the web scraping script to collect fresh reviews:

python web_scraping.py

**Dependencies**





Python 3.8+



Libraries: pandas, numpy, matplotlib, seaborn, textblob, scikit-learn, scipy, google-play-scraper

**Authors**



Annanya Singh (Regd. No.: 23375010)

Hrithik Dineshan (Regd. No.: 20384308)

