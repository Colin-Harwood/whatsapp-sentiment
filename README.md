<h1>A deep neural network that relies on tensorflow, python, numpy and pandas utilising twitter sentiment data to train a model that can predict the sentiment of a whatsapp chat log fed into it.</h1>

<h2>The model is trained using tensorflow keras to create the neural network along with scikit learn for preprocessing, matplotlib for visualising the neural networks performance and pandas to format the data into a data frame.</h2>

<h3>
  The neural network uses NLP to tokenise all of the words in the training data - the twitter sentiment csv - pad the data, preprocess the data, then split the training and test data into seperate dataframes that can be fed into the model.</br>
  After this the model is created, and the data is fit to the models and the models are trained. The best model as well as the final model are both saved. </br>
  Lastly the final results are printed out as well as the training & validation loss values, and the training & validation accuracy values are plotted with matplotlib.
</h3>

<h3>
  The whatsappCheck.py is what handles taking the whatsapp chat log and preprocessing as well as tokenising in the same method as the model training with some additional regex to remove unnecesary info.</br>
  The whatsapp chats are also split into sentences for consistency so the final average of all of the individual sentences is added up and average of the overall sentiment of the whatsapp is given out!
</h3>
