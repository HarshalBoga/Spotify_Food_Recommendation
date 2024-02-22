# E15_Streamlit

Music and Food Recommendation App

Features:

Retrieve user's top songs and calculate audio feature similarity from the Spotify api.
Display matching music preferences and their similarity percentages.
Get real-time weather information based on the user's city.
Recommend vegan food and beverage options if the user is vegan.
Provide food and beverage recommendations at different levels:
Level 1: General Top Selling Items
Level 2: Perfect Things to Eat and Drink based on Vibe
Level 3: Advanced Personalized Recommendations
Generate QR codes for general information.


Spotify API for user's top songs and audio features.
Weather API for real-time weather data based on the user's city.
Pre-defined vegan food and beverage menu data.
Preprocessing
Data preprocessing steps include data cleaning, feature engineering, and data normalization. Missing values are handled, and data is normalized for similarity calculation.

Music Preference Analysis
The app matches user preferences with audio features of the artist's top songs and calculates similarity percentages. The top matching features are displayed to the user.

Weather Information
Weather information is retrieved based on the user's provided city name using a Weather API. It includes temperature, humidity, climate, and current weather conditions.

Vegan Food and Beverage Menu
If the user indicates they are vegan, the app displays a vegan food and beverage menu with various options and their average calories and prices.

Recommendation Levels
The app provides food and beverage recommendations at different levels:

Level 1: General Top Selling Items

Level 2: Perfect Things to Eat and Drink based on Vibe

Level 3: Advanced Personalized Recommendations

QR Code Generation
QR codes are generated for general user information. They can be scanned and shared for easy access to recommendations.

Implementation of Predictive Models
The app includes machine learning models for similarity calculation and recommendation generation. The models are trained on preprocessed data and evaluated for performance.



