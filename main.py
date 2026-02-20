import nltk
import numpy as np
import random
import json
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load JSON data from file
with open("p:/Users/lolon/OneDrive/Έγγραφα/Αρχεία Πάνος/programming/python/myChatBot/intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)


# Data preprocessing
words = []
classes = []
documents = []
ignore_words = ['?', '!', '@']

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
    classes.append(intent["tag"])

# Lemmatize words
words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]))
classes = sorted(set(classes))

# Convert text data into numerical format
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [1 if w in [lemmatizer.lemmatize(word.lower()) for word in doc[0]] else 0 for w in words]
    output_row = list(output_empty)                        # Create a copy of output_empty
    output_row[classes.index(doc[1])] = 1                  # output_row is set to 1, meaning this sentence belongs to that class
    training.append([bag, output_row])                     # Store the bag of words and the output_row together in the training list as a pair

# Shuffle and convert to numpy arrays
random.shuffle(training)
training = np.array(training, dtype=object)

X_train = np.array(list(training[:, 0]))
y_train = np.array(list(training[:, 1]))

# Define the neural network
model = Sequential([
    Dense(16, activation='relu', input_shape=(len(X_train[0]),)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(len(y_train[0]), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=300, batch_size=8, verbose=1)

# Helper functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    return np.array([1 if w in sentence_words else 0 for w in words])

def load_feedback():
    feedback_responses = {}
    try:
        with open("feedback.json", "r", encoding = "utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                feedback_responses[entry["question"].lower()] = entry["corrected_response"]
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        print("Error : feedback.json is corrupted or incorrectly formatted.")

    return feedback_responses


feedback_responses = load_feedback() # Load feedback at the start


def predict_response(text):
    text_lower = text.lower().strip()

    # Check if the question exists in feedback.json
    if text_lower in feedback_responses:
        return feedback_responses[text_lower]
    

    print(f"User input: {text}")  # Debug print
    bow_vector = bow(text, words)
    print(f"BOW vector: {bow_vector}")  # Debug print
    res = model.predict(np.array([bow_vector]), verbose=0)[0]
    print(f"Prediction: {res}")  # Debug print
    max_index = np.argmax(res)

    if res[max_index] > 0.3:
        tag = classes[max_index]
        for intent in data["intents"]:
            if intent["tag"] == tag:
                response =  random.choice(intent["responses"])
                print("ChatBot:", response)

                # Ask for feedback
                feedback = input("Was this response helpful? (yes/no): ").strip().lower()
                while feedback != "yes" and feedback != "no":
                    feedback = input("Was this response helpful? (yes/no): ").strip().lower()
                    if feedback == "no":
                            corrected_response = input("What should the response be? ")
                            save_feedback(text, corrected_response)
                            feedback_responses[text_lower] = corrected_response # Update memory
                            return response
    else:
        return "I'm sorry, I don't understand the question."

def save_feedback(question, corrected_response):
    feedback_data = {"question": question.strip().lower(), "corrected_response": corrected_response.strip()}

    # Append feedback to a JSON file
    with open("feedback.json", "a", encoding = "utf-8") as f:
        json.dump(feedback_data, f)
        f.write("\n") # New line for each feedback entry 

        print("Thank you! Your feedback has been saved.")
 
# Run the chatbot
while True:
    user_input = input("Your question: ")
    if user_input.lower() == "exit":
        print("ChatBot: Goodbye!")
        break
    print("ChatBot:", predict_response(user_input))
