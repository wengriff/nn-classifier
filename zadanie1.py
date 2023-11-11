# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import seaborn as sns


# %%
# Allow printing more columns
pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

# %%
# Do not show pandas warnings
pd.set_option('mode.chained_assignment', None)

# %%
df = pd.read_csv('./zadanie1_dataset.csv')

# %%
# rpint the number of rows and columns
print("Number of rows and columns: ", df.shape)

# %%
# Read the columns and print them
columns = df.columns
print(columns)

# %%
# Print min and max values of columns before removing outliers
print("*"*100, "Before removing outliers", "*"*100)
print("-"*10, "Min", "-"*10)
print(df.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(df.max(numeric_only=True))

# %%
# Remove unnecessary columns
# df = df.drop(columns=['number_of_artists', 'url', 'name', 'genres', 'filtered_genres', 'top_genre'])
df = df.drop(columns=['url', 'name', 'genres', 'filtered_genres', 'top_genre'])



# %%
# Print min and max values of columns before removing outliers
print("*"*100, "Before removing outliers", "*"*100)
print("-"*10, "Min", "-"*10)
print(df.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(df.max(numeric_only=True))

# %%
# Deal with outliers - Remove outliers
df = df[(df['danceability'] >= 0) & (df['danceability'] <= 1)]
df = df[(df['loudness'] >= -60) & (df['loudness'] <= 0)]

# %%
# Print min and max values of columns before removing outliers
print("*"*100, "Before removing outliers", "*"*100)
print("-"*10, "Min", "-"*10)
print(df.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(df.max(numeric_only=True))

# %%
df = df[df['duration_ms'] > 0]

# %%
null_count = df.isnull().sum()
print("Number of null values in each column:")
print(null_count)

# %%
df = df.dropna(subset=['popularity'])
df = df.dropna(subset=['number_of_artists'])

# %%
non_numeric_columns = df.select_dtypes(include=['object']).columns
print("Non-numeric columns:")
print(non_numeric_columns)

# %%
df = pd.get_dummies(df, columns=['emotion'], prefix=['emotion'])
df['explicit'] = df['explicit'].astype(int)

# %%
print(df.head())

# %%
# Split the data into input and output
X = df.drop(columns=['emotion_calm', 'emotion_energetic', 'emotion_happy', 'emotion_sad'])
y = df[['emotion_calm', 'emotion_energetic', 'emotion_happy', 'emotion_sad']].astype(int)

# %%
y

# %%
# Split dataset into train, valid and test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=69)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, shuffle=True, test_size=0.5, random_state=69)

# %%
# Print dataset shapes
print("*"*100, "Dataset shapes", "*"*100)
print(f"X_train: {X_train.shape}")
print(f"X_valid: {X_valid.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_valid: {y_valid.shape}")
print(f"y_test: {y_test.shape}")

# %%
X_train.hist(figsize=(12, 10), bins=50)
plt.suptitle('Feature Distributions Before Scaling')
plt.show()

# %%
# Print min and max values of columns
print("*"*100, "Before scaling/standardizing", "*"*100)
print("-"*10, "Min", "-"*10)
print(X_train.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(X_train.max(numeric_only=True))

# %%
# Standardize data
scaler = StandardScaler()
# !!!!!
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# %%
# Convert numpy arrays to pandas DataFrames
X_train = pd.DataFrame(X_train, columns=X.columns)
X_valid = pd.DataFrame(X_valid, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# %%
# after standardization
X_train.hist(figsize=(12, 10), bins=50)
plt.suptitle('Feature Distributions After Scaling')
plt.show()

# %%
count_no_lyrics = df[df['explicit'] == False].shape[0]
print(f"Number of rows without lyrics: {count_no_lyrics}")

# %%
# Print min and max values of columns
print("*"*100, "After scaling/standardizing", "*"*100)
print("-"*10, "Min", "-"*10)
print(X_train.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(X_train.max(numeric_only=True))

# %%
unique_combinations = y_train.drop_duplicates().shape[0]
random_accuracy = 1 / unique_combinations

print(f"Random accuracy based on unique combinations: {random_accuracy}")


# %%
clf = MLPClassifier(
    hidden_layer_sizes=(128, 128, 128),
    random_state=1,
    validation_fraction=0.2,
    early_stopping=True,
    learning_rate='adaptive',
    learning_rate_init=0.0005,
).fit(X_train, y_train)

# %%
# Predict on train set
y_pred = clf.predict(X_train)

# %%
# Convert one-hot encoded y_test and y_pred to integer labels
y_train_int = np.argmax(y_train.values, axis=1)
y_pred_int = np.argmax(y_pred, axis=1)

# %%
# Calculate accuracy
print('MLP accuracy on train set:', accuracy_score(y_train_int, y_pred_int))

# Generate confusion matrix
cm_train = confusion_matrix(y_train_int, y_pred_int)

# %%
# Predict on test set
y_pred = clf.predict(X_test)

# %%
# Convert one-hot encoded y_test and y_pred to integer labels
y_test_int = np.argmax(y_test.values, axis=1)
y_pred_int = np.argmax(y_pred, axis=1)

# %%
# Calculate accuracy
print('MLP accuracy on test set:', accuracy_score(y_test_int, y_pred_int))

# Generate confusion matrix
cm_test = confusion_matrix(y_test_int, y_pred_int)

# %%
# Get class names for the one-hot encoded 'emotion' column
class_names = [col for col in df.columns if col.startswith('emotion_')]

# Optionally, you can remove the 'emotion_' prefix to get the original emotion names
class_names = [col.replace('emotion_', '') for col in class_names]


# %%
disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on train set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()

# %%
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on test set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()

# %%
original_dataset_path = './zadanie1_dataset.csv'
df_original = pd.read_csv(original_dataset_path)

# Display the first few rows of the original dataset to understand its structure
df_original.head()

# %%
# Overview of data types and null values
print(df_original.info())

# %%
df.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(df.corr(), annot=True)
plt.show()

# %%
# Energy and Loudness: High positive correlation (0.87) suggests tracks that are louder are generally more energetic.
# The relationship between acousticness and energy shows a moderate negative correlation of −0.80 
# This suggests that tracks with higher acousticness generally tend to have lower energy levels.
# calm emotions are negatively correlated with energy and loudness

# %%
# print df columns
print(df.columns)

# %%
# Grouping the data by 'emotion' and calculating the average 'popularity' for each group
emotion_popularity = df_original.groupby('emotion')['popularity'].mean().sort_values(ascending=False)

# Creating the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=emotion_popularity.index, y=emotion_popularity.values, palette='viridis')
plt.title('Average Popularity by Emotion')
plt.xlabel('Emotion')
plt.ylabel('Average Popularity')
plt.show()


# %%
# The more happy and energetic the song the popular it is 

# %%
# Calculate mean, median, and count for each emotion
emotion_stats = df_original.groupby('emotion')['popularity'].agg(['mean', 'median', 'count']).sort_values(by='mean', ascending=False)

print(emotion_stats)


# %%
from scipy.stats import gaussian_kde
# Select specific columns (assuming these are the columns you're interested in)
selected_columns = ['danceability', 'energy', 'loudness', 'speechiness', 
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 
                    'tempo', 'duration_ms', 'popularity']

# Initialize the matplotlib figure
fig, axs = plt.subplots(4, 3, figsize=(15, 10))

# Reshape axs to 1-D array
axs = axs.ravel()

# Generate density plots
for i, feature in enumerate(selected_columns):
    data = df[feature].dropna()  # Drop NaN values if any
    density = gaussian_kde(data)
    xs = np.linspace(min(data), max(data), 200)
    density.covariance_factor = lambda: 0.25
    density._compute_covariance()
    axs[i].plot(xs, density(xs), label=f"{feature}")
    axs[i].fill_between(xs, density(xs), alpha=0.5)
    axs[i].set_title(f"{feature}")
    # axs[i].set_xlabel(feature)
    # axs[i].set_ylabel('Density')

# Set a title for the entire grid
fig.suptitle('Density Plots of Selected Features', fontsize=32)
plt.tight_layout()
plt.show()

# %%
# Create a bar chart for all 'top_genre' vs 'popularity'
# Group the data by 'top_genre' and calculate the average 'popularity' for each group
genre_popularity_all = df_original.groupby('genre')['popularity'].mean().sort_values(ascending=False)

# Creating the bar chart using only Matplotlib
plt.figure(figsize=(14, 12))
plt.barh(genre_popularity_all.index, genre_popularity_all.values, color='orange')
plt.title('Average Popularity by All Top Genres')
plt.xlabel('Average Popularity')
plt.ylabel('Top Genre')
plt.show()


# %%
# The j-pop is the most popular and country is the least popular genre. 
# The varying levels of average popularity across genres suggest that 'top_genre' could be a significant 
# categorical predictor for modeling or understanding track success.

# %%
df_cleaned =  df_original[(df_original['danceability'] >= 0) & (df_original['danceability'] <= 1)]

# %%
plt.figure(figsize=(12, 8))
sns.scatterplot(x='danceability', y='energy', hue='emotion', data=df_cleaned, palette='Blues', alpha=0.7)
plt.title('Scatter Plot of Danceability vs Energy, Color-coded by Emotion')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
# ChatGPT

# %%
# Happy songs are often more danceable and energetic than sad songs.

# %%
# Find the genres that are most likely to be explicit
most_explicit_genres = df_cleaned.groupby('top_genre')['explicit'].mean().sort_values(ascending=False).head(10)

# %%
# Plotting the genres that are most likely to be explicit
plt.figure(figsize=(12, 6))
sns.barplot(x=most_explicit_genres.index, y=most_explicit_genres.values, palette="viridis")
plt.title('Genres Most Likely to be Explicit')
plt.xlabel('Genre')
plt.ylabel('Likelihood of Being Explicit')
plt.xticks(rotation=30)
plt.show()

# %%
## 3rd task
# USE ADAM or SDG with moment in order to get out of local minima
# USE RELU
# High Rate of learning might not be able to find the global minima, low will not get u there
# velka siet = high learning rate
# mala siet - low learning rate
# zastavovat trenovanie podla chyby, validation error
# architerkura 
# model add dense 4 activation softmax, 
# multiclass cross entropy
# 30+ batch size


# %%
# print the columns
print(df_original.columns)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot
sns.scatterplot(x='speechiness', y='instrumentalness', data=df)
plt.title('Scatter Plot of Speechiness vs Instrumentalness')
plt.xlabel('Speechiness')
plt.ylabel('Instrumentalness')
plt.show()


# %%
plt.scatter(df['loudness'], df['acousticness'])
plt.xlabel('Loudness')
plt.ylabel('Acousticness')
plt.title('Loudness vs Acousticness')
plt.show()


