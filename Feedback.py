from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

input_data = []

data = ["Wow Nice This Project is really cool!", "I guess it is ok", "worst thing ever", "you suck", "Its pretty good", "ok", "its not good but not bad", "awesome! pretty good ig", "it is very bad!", "TRASH!", "cool", "hate this"]
vectorizer = TfidfVectorizer()
input_data = vectorizer.fit_transform(data)

#print(input_data)


#0 = good, 1 = ok, 2 = bad
output_data = [0, 1, 2, 2, 0, 1, 1, 0, 2, 2, 0, 2]

model = svm.SVC()

model.fit(input_data, output_data)

feedback = input("What is your feedback on this project?\n")

test_data = ["Cool", "Why did you even bother.", "awesome!"]
test_data.append(feedback)

e = model.predict(vectorizer.transform(test_data))
print(e)

if e[3] == 0:
  print("Thanks for the nice feedback! your feedback was [POSITIVE]")
if e[3] == 1:
  print("Thanks for the feedback! your feedback [OK]")
if e[3] == 2:
  print("Y u so mean :( [NEGATIVE]")