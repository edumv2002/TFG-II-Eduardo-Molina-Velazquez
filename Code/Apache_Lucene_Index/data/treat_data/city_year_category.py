# this file will analyzre cambridge_2014_streets.csv, parse its content and create a dictionary with the following structure:
# {word: number of times it appears in the file}

import mysql.connector
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Connect to your MySQL database
db_connection = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="eduardomv",
    password="*****",
    database="participatory_budgeting"
)

cursor = db_connection.cursor()

# List of datasets (cities and years)
city = input('Enter the dataset you want to analyze: (Cambridge, Oxford, NewYorkCity): ')
year = input('Enter the year you want to analyze: (2014, 2015, 2016, 2017): ')
dataset = city + '-' + year
category = input('Enter the category you want to analyze: ')
# Dictionary to store proposal titles for each dataset
word_count = {}

query = f"SELECT description FROM items WHERE dataset = '{dataset}' and category = '{category}'"
cursor.execute(query)

# Fetch all the proposal titles
proposals = [row[0] for row in cursor.fetchall()]
print('la longitud de las propuestas es',len(proposals))
# Close the connection
cursor.close()
db_connection.close()
tokens_list = []
for p in proposals:
    text = p.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove punctuation
    tokens = [word for word in tokens if word.isalpha()]
    # we concatenate the list of tokens with the list of tokens of the current proposal
    tokens_list += tokens
    
# from the token list, I need to remove words that are not useful for the analysis
# I will remove the stopwords
tokens_list = [word for word in tokens_list if not word in stop_words]
# now we have a list of words, we can count the number of times each word appears
for word in tokens_list:
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1

# Now we print an ordered dictionary based on the number of times each word appears
word_count = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True))
city = city.lower()
# Now we save the dictionary in a file in a legible format. I need each key-value pair in a new line
with open(f'{city}/{year}/{city}_{year}_{category}.txt', 'w') as f:
    for key in word_count.keys():
        f.write(f'{key}: {word_count[key]}\n')
    print('The file has been created')
    
