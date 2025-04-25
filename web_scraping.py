# Web-Scrapping to obtain the data

import subprocess
import sys
import time
from datetime import datetime
import pandas as pd
import os

class ReviewDataHandler:
    def _init_(self, local_file_path=None):
        """
        Initialize the handler with either a local file path or default to None for scraping
        """
        self.local_file_path = local_file_path
        self.df = None

    def load_local_file(self):
        """
        Load reviews from a local CSV file
        """
        try:
            if not os.path.exists(self.local_file_path):
                raise FileNotFoundError(f"File not found: {self.local_file_path}")
                
            self.df = pd.read_csv(self.local_file_path)
            
            # Verify required columns
            required_columns = [
                'Review ID', 'User Name', 'Review Text', 'Rating',
                'Likes', 'App Version Reviewed', 'Review Date', 'Current App Version'
            ]
            
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Convert timestamp to datetime if it isn't already
            self.df['Review Date'] = pd.to_datetime(self.df['Review Date'])
            
            return self.df
            
        except Exception as e:
            print(f"Error loading local file: {str(e)}")
            return None

    def scrape_reviews(self, app_package_name, num_reviews=500):
        """
        Scrape reviews from Google Play Store with error handling and rate limiting.
        """
        try:
            from google_play_scraper import reviews, Sort
            
            # Add delay between batches to avoid rate limiting
            batch_size = 250
            all_reviews = []
            
            for i in range(0, num_reviews, batch_size):
                try:
                    print(f"Fetching reviews {i+1} to {min(i+batch_size, num_reviews)}...")
                    review_batch, continuation_token = reviews(
                        app_package_name,
                        lang='en',
                        country='us',
                        sort=Sort.NEWEST,
                        count=min(batch_size, num_reviews-i)
                    )
                    all_reviews.extend(review_batch)
                    time.sleep(2)  # Add delay between batches
                    
                except Exception as e:
                    print(f"Error fetching batch: {str(e)}")
                    break
            
            if not all_reviews:
                print("No reviews were fetched.")
                return None
                
            # Convert to DataFrame
            self.df = pd.DataFrame(all_reviews)
            
            # Select and rename columns for clarity
            columns = {
                'reviewId': 'Review ID',
                'userName': 'User Name',
                'content': 'Review Text',
                'score': 'Rating',
                'thumbsUpCount': 'Likes',
                'reviewCreatedVersion': 'App Version Reviewed',
                'at': 'Review Date',
                'appVersion': 'Current App Version'
            }
            
            self.df = self.df[list(columns.keys())].rename(columns=columns)
            
            # Convert timestamp to readable date
            self.df['Review Date'] = pd.to_datetime(self.df['Review Date'])
            
            return self.df
            
        except Exception as e:
            print(f"Error in scrape_reviews: {str(e)}")
            return None

    def save_to_csv(self, output_file=None):
        """
        Save the current DataFrame to a CSV file
        """
        if self.df is None:
            print("No data to save.")
            return False
            
        try:
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f'netflix_reviews_{timestamp}.csv'
                
            self.df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\nSuccessfully saved {len(self.df)} reviews to '{output_file}'")
            return True
            
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")
            return False

def main():
    # For loading from local file:
    local_handler = ReviewDataHandler(local_file_path='path/to/your/netflix_reviews.csv')
    df_local = local_handler.load_local_file()
    
    # For scraping new data:
    scrape_handler = ReviewDataHandler()
    app_package_name = 'com.netflix.mediaclient'
    df_scraped = scrape_handler.scrape_reviews(app_package_name, num_reviews=10000)
    scrape_handler.save_to_csv()

if _name_ == "_main_":
    main()

df = pd.read_csv("netflix_reviews.csv") # File path
