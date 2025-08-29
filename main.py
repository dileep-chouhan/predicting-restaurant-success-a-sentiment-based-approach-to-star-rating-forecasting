import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Download VADER lexicon if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_reviews = 200
data = {
    'StarRating': np.random.choice([1, 2, 3, 4, 5], size=num_reviews),
    'ReviewText': [f'Review {i}: This is a sample review.' for i in range(num_reviews)]
}
# Add some variability to the reviews
for i in range(num_reviews):
    if data['StarRating'][i] <= 2:
        data['ReviewText'][i] += ' The food was terrible and the service was slow.'
    elif data['StarRating'][i] == 3:
        data['ReviewText'][i] += ' The food was okay, but nothing special.'
    elif data['StarRating'][i] >= 4:
        data['ReviewText'][i] += ' I had a great experience! The food was delicious and the service was excellent.'
df = pd.DataFrame(data)
# --- 2. Sentiment Analysis ---
analyzer = SentimentIntensityAnalyzer()
df['SentimentScores'] = df['ReviewText'].apply(lambda review: analyzer.polarity_scores(review)['compound'])
# --- 3. Analysis ---
# Group by star rating and calculate average sentiment score
average_sentiment = df.groupby('StarRating')['SentimentScores'].mean()
# --- 4. Visualization ---
plt.figure(figsize=(10, 6))
sns.barplot(x=average_sentiment.index, y=average_sentiment.values)
plt.title('Average Sentiment Score by Star Rating')
plt.xlabel('Star Rating')
plt.ylabel('Average Sentiment Score')
plt.grid(True)
plt.tight_layout()
# Save the plot to a file
output_filename = 'sentiment_by_rating.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
# --- 5.  Further Analysis (Example: Correlation) ---
correlation = df['StarRating'].corr(df['SentimentScores'])
print(f"\nCorrelation between Star Rating and Sentiment Score: {correlation}")
# --- 6. (Optional) Predictive Modeling (Simple example - Linear Regression)---
#Note: This is a highly simplified example and would require more sophisticated techniques for real-world application.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = df[['SentimentScores']]
y = df['StarRating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print(f"\nLinear Regression Model R-squared: {model.score(X_test, y_test)}")