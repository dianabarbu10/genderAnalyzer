from sklearn import tree

#height, #weight, #shoe size
x=[[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,42]]
y=['male','female','female','female','male','male','male','male']

clf=tree.DecisionTreeClassifier()
clf=clf.fit(x,y)
predictionRandom = clf.predict([[190,80,44]])
print("Random prediction is: "+str(predictionRandom))

predictionMe = clf.predict([[160,50,37]])
print("Me prediction should be female and is: "+str(predictionMe))

predictionDad = clf.predict([[175,80,42]])
print("Mada prediction should be male and is: "+str(predictionDad))

predictionLaurasIubi = clf.predict([[165,57,39]])
print("Laura's iubi prediction should be male and is: "+str(predictionLaurasIubi))


predictionGicuTest = clf.predict([[175,150,37]])
print("Gicu's test prediction should be female and is: "+str(predictionGicuTest))