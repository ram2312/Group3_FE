

# ML-Code

# All the Required Libraries
# These Libraries deals with Data Preprocessing

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import streamlit as st

# These Libraries deal with model importation training and Prediction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

# Loading all the sub datasets to be used
sub_set1 = pd.read_csv('Credits.csv')
sub_set2 = pd.read_csv('Movies.csv')
sub_set3 = pd.read_csv('Ratings.csv')

"""**Viewing The Initial  Rows of the Sub Datasets**"""

# Viewing the Initial 5 rows of the  Movie credits
sub_set1.head()

# Viewing the Initial 5 rows of the  Movie dataset
sub_set2.head()

# Viewing the Initial 5 rows of the  Movie credits using a Reader Object
reader = Reader()
sub_set3.head()

"""**Exploratory Data Analysis**"""

# Renaming the Columns of  credits  dataset inorder to merge it with Movie  using common column id
sub_set1.columns = ['id','tile','cast','crew']
sub_set2= sub_set2.merge(sub_set1,on='id')

# viewing the improved movies dataset
sub_set2.head()

"""1.Data Visualization and Checking The Central Tendancy"""

#check the  Mean and 90th percentile of the Movie dataset
Mean= sub_set2['vote_average'].mean()
print(Mean)
per= sub_set2['vote_count'].quantile(0.9)
print(per)

# Creating a bar graph
fig, ax = plt.subplots(figsize=(10, 10))
ax.bar(['Mean of vote averages (C)', '90th percentile of vote counts (m)'], [Mean, per], color=['b', 'g'])
ax.set_xlabel('Metric')
ax.set_ylabel('Value')
ax.set_title('Bar graph of mean of vote averages Mean and 90th percentile of vote counts per')
ax.grid(True)

# Display the plot in Streamlit app
st.pyplot(fig)

# Creating a pie chart
labels = ['Mean of vote averages (C)', '90th percentile of vote counts (m)']
sizes = [Mean, per]
colors = ['b', 'g']
explode = (0.1, 0)

fig, ax = plt.subplots(figsize=(10, 10))
ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
ax.set_title('Pie chart of mean of vote averages (Mean) and 90th percentile of vote counts (Per)')
ax.axis('equal') 

# Display the pie chart in Streamlit app
st.pyplot(fig)

"""**2.Demographic Filtering**

"""

# getting  the shape of  sub movie dataset that is greater than or equal  Mean
q_movies = sub_set2.copy().loc[sub_set2['vote_count'] >= Mean]
q_movies.shape

# Defining a function for weighted rating based on IMDB formula
def weighted_rating(x, m=per, C=Mean):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)

pop = sub_set2.sort_values('popularity', ascending=False)

# Creating a horizontal bar plot
fig, ax = plt.subplots(figsize=(12, 4))

ax.barh(pop['title'].head(6), pop['popularity'].head(6), align='center', color='skyblue')
ax.invert_yaxis()
ax.set_xlabel("Popularity")
ax.set_title("Popular Movies")

# Display the horizontal bar plot in Streamlit app
st.pyplot(fig)

"""**Content Based Filtering**

A recommendation method called content-based filtering makes recommendations to a consumer based on the qualities of products they have previously liked. Content-based filtering algorithms examine attributes that users have found enjoyable in movies, including storyline keywords, actors, directors, and genre, in order to spot trends and preferences when it comes to movie suggestion. The algorithm suggests comparable films that the viewer would probably like based on these trends.

The content-based filtering algorithm, for example, will give recommendations for more comedic films precedence if the user has a history of watching and rating comedies well. This is because the system recognizes the user's apparent liking for humor and lighter amusement.

In addition, the algorithm will recommend movies with actors or directors that the user has indicated they enjoy watching.

Personalized movie suggestions and individualised tastes may be achieved with the use of content-based filtering. The technology can efficiently direct users toward films that match their interests by examining user preferences and seeing trends in their previous selections.


"""

# Display the overview of the first few movies
sub_set2['overview'].head()

# Text Vectorization
tfidf = TfidfVectorizer(stop_words = 'english')

sub_set2['overview'] = sub_set2['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(sub_set2['overview'])

tfidf_matrix.shape

"""**Text Vectorization**

In machine learning, text vectorization is the process of converting text input into numerical vectors. This is significant since machine learning techniques can only handle numerical data. These two broad categories of text vectorization techniques are:


Techniques that rely on counts: These techniques simply count how many times each word appears in a document. Two techniques that may be applied for this are TF-IDF and Bag-of-words (BoW).

Word embedding techniques include: Rather than focusing only on word counts, these approaches aim to capture the meaning of words and their relationships with one another. There are two ways to accomplish this: Word2Vec and GloVe.

Machine learning applications such as sentiment analysis, topic modeling, and natural language processing (NLP) rely on it.


"""

# Import TfidfVectorizer for text vectorization
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(sub_set2.index, index = sub_set2['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies

def get_recommendations(title, cosine_sim = cosine_sim):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return sub_set2['title'].iloc[movie_indices]

# Getting Recommendation
get_recommendations('Minions')

get_recommendations('The Avengers')

"""Credits, Genres and Keywords Based Recommender"""

# Parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    sub_set2[feature] = sub_set2[feature].apply(literal_eval)

# Define functions to extract directors  from features
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Define functions to get list of names from features

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]

        if len(names) > 3:
            names = names[:3]
        return names


    return []

sub_set2['director'] = sub_set2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    sub_set2[feature] = sub_set2[feature].apply(get_list)

# Print the new features of the first 3 films
sub_set2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ","")) for i in x]

    else:

        if isinstance(x , str):
            return str.lower(x.replace(" ",""))
        else:
            return ''

# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    sub_set2[feature] =sub_set2[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
sub_set2['soup'] = sub_set2.apply(create_soup, axis=1)

# Import CountVectorizer and create the count matrix
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(sub_set2['soup'])

# Compute the Cosine Similarity matrix based on the count_matrix
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset index of our main DataFrame and construct reverse mapping as before
sub_set2 = sub_set2.reset_index()
indices = pd.Series(sub_set2.index, index=sub_set2['title'])

# # Prediction Corner
# Movie=input("Enter Movie Name to get Other Recommendations:")
# get_recommendations(Movie, cosine_sim2)

"""Collaborative Filtering"""

data = Dataset.load_from_df(sub_set3[['userId', 'movieId', 'rating']], reader)

svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'])

trainset = data.build_full_trainset()
svd.fit(trainset)

sub_set3[sub_set3['userId'] == 1]

svd.predict(1, 302, 3)



if __name__ == "__main__":
    st.title("Movie Recommendation and Rating System")
    
    # Add Streamlit components as needed
    # For example, you can create input fields and display results

    movie_input = st.text_input("Enter Movie Name to get Other Recommendations:")
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(movie_input, cosine_sim2)
        st.write("Recommended Movies:")
        st.write(recommendations)
