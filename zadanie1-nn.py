# %%
import pandas as pd
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.callbacks import EarlyStopping

# %%
df = pd.read_csv('./zadanie1_dataset.csv')

# %%
# Remove unnecessary columns
df = df.drop(columns=['url', 'name', 'genres', 'filtered_genres', 'top_genre'])

# %%
# Deal with outliers - Remove outliers
df = df[(df['danceability'] >= 0) & (df['danceability'] <= 1)]
df = df[(df['loudness'] >= -60) & (df['loudness'] <= 0)]
# df = df[(df['duration_ms'] > 0) & (df['duration_ms'] < 1000000)]
df = df.dropna(subset=['popularity'])
df = df.dropna(subset=['number_of_artists'])

# %%
# Encode
df['explicit'] = df['explicit'].astype(int)

# %%
# Split the data into input and output
X = df.drop(columns=['emotion'])
y = df['emotion']

# %%
y = pd.get_dummies(df['emotion'])

# Rename the columns to match your desired output
y.columns = ['calm', 'energetic', 'happy', 'sad']

# %%
# Split dataset into train, valid and test
X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5, random_state=42)

# %%
# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# %%
# Convert numpy arrays to pandas DataFrames
X_train = pd.DataFrame(X_train, columns=X.columns)
X_valid = pd.DataFrame(X_valid, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# %%
# Train MLP overtrained_model in Keras
# model = Sequential()
# model.add(Dense(1024, input_dim=X_train.shape[1], activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(4, activation='softmax'))

# model = Sequential()
# model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(4, activation='softmax'))

# model = Sequential()
# model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(4, activation='softmax'))

# model = Sequential()
# model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(4, activation='softmax'))

model = Sequential()
model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(4, activation='softmax'))

# %%
# model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.00002), metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# %%
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# %%
history = model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), epochs=1000, batch_size=32, callbacks=[early_stopping])

# %%
# evaluate the overtrained_model on the training set
train_scores = model.evaluate(X_train, y_train, verbose=0)

# %%
print("*"*100, "Test accuracy", "*"*100)
print(f"Train accuracy: {train_scores[1]:.4f}")

# %%
# Evaluate the overtrained_model
test_scores = model.evaluate(X_test, y_test, verbose=0)

# %%
print("*"*100, "Test accuracy", "*"*100)
print(f"Test accuracy: {test_scores[1]:.4f}")

# %%
# Plot confusion matrix
y_pred = model.predict(X_test)
y_pred = (y_pred >= 0.25)

# %%
y_test_int = np.argmax(y_test.values, axis=1)
y_pred_int = np.argmax(y_pred, axis=1)

# %%
# Assuming true_labels and predicted_labels are already defined
cm = confusion_matrix(y_test_int, y_pred_int)

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(cm, cmap=plt.cm.Blues)
fig.colorbar(cax)

ax.set_xticks(np.arange(4))
ax.set_yticks(np.arange(4))
ax.set_xticklabels(['calm', 'energetic', 'happy', 'sad'])
ax.set_yticklabels(['calm', 'energetic', 'happy', 'sad'])

plt.xlabel('Predicted')
plt.ylabel('True')

# Loop over data dimensions and create text annotations.
for i in range(4):
    for j in range(4):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="red")

plt.show()

# %%
# Plot loss and accuracy
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()


# %%
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.show()


