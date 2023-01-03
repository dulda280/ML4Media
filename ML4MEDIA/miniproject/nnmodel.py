import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
# load dataset


letterData = pd.read_csv('E:\MED-local\MED7\ML4Media\ML4MEDIA\miniproject\data\letterrecognitiondatacsv.csv')

#[x-box, y-box, width, height, onpix, x-bar, y-bar, x2bar, y2bar, x2ybr, xy2br, x-edg, xedgvy, y-edg, yedgvx]
features = []
letters = []
dataframe = []

#get features and labels (letters) from dataset and get dataframe set up properly
for i in range(0, len(letterData)):
    temp = letterData.iloc[i, 0]
    temp = temp.split(';')
    newTemp = [int(temp[i]) for i in range(1, len(temp))]
    letter = temp[0]
    features.append(newTemp)
    letters.append(letter)
    tempDF = newTemp
    tempDF.insert(len(newTemp), temp[0])
    dataframe.append(tempDF)
print(dataframe[0])

df = pd.DataFrame(dataframe, columns=['x-box', 'y-box', 'width', 'height', 'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-edge', 'yedgevx', 'letter'])
df3 = pd.DataFrame(dataframe, columns=['x-box', 'y-box', 'width', 'height', 'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-edge', 'yedgevx', 'letter'])
df3.rename(columns=df.iloc[0]).drop(df.index[0])

dataset = df3.values
X = dataset[:,0:15].astype(float)
Y = dataset[:,16]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=15, activation='relu'))
	model.add(Dense(26, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=5, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))