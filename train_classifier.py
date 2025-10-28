import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# stratify -> keep same proportion of all our different labels (1/3 each)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(X_train, y_train)

model.predict(X_test)
y_predict = model.predict(X_test)

score = accuracy_score(y_predict, y_test) # probability, so 0 to 1
print('{}% of samples were classified correctly!'.format(score*100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()