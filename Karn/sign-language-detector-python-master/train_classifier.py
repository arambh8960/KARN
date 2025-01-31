# import pickle

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np


# data_dict = pickle.load(open('./data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# model = RandomForestClassifier()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly !'.format(score * 100))

# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()


# import pickle

# import numpy as np

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# data_dict = pickle.load(open('./data.pickle','rb'))

# data =np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# model = RandomForestClassifier()

# model.fit(x_train, y_train)
# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print("{}% of samples were classified correctly !".format(score * 100))

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

# Check the length of each data sample
consistent_length = len(data[0])  # Assuming the first sample has the intended length
filtered_data = []
filtered_labels = []

for sample, label in zip(data, labels):
    if len(sample) == consistent_length:  # Only keep samples with the correct length
        filtered_data.append(sample)
        filtered_labels.append(label)
    else:
        print(f"Filtered out inconsistent sample with length {len(sample)}")

# Convert to numpy arrays
filtered_data = np.asarray(filtered_data)
filtered_labels = np.asarray(filtered_labels)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    filtered_data, filtered_labels, test_size=0.2, shuffle=True, stratify=filtered_labels
)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"{score * 100}% of samples were classified correctly!")

f = open("model.p", "wb")
pickle.dump({"model":model}, f)
f.close()
