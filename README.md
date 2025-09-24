# Sentiment Analysis Classifier with DistilBERT and Scikit-Learn Classifier
## About
This app predicts the sentiment of movie reviews. It also gives you the closest review in similarty search. You can use the deployed version of this [app]()
- Model used: [disitillBERT](https://huggingface.co/distilbert/distilbert-base-uncased) 
- Dataset used: [dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) , License: unknown
- Framework and deploy: [streamlit + streamlit]()

## Feature
- Predict movie review sentiment (Positive/Negative).
- Show the closets reviews from the dataset and the distance between them.



# Usage 
1. Enter a review of a movie, for example: (more sample input available at `sample_input.txt`)
Komentar Positif:
"I absolutely loved this movie! The acting was superb and the storyline was captivating."
"This is one of the best films I've seen in a long time. Amazing visuals and great performances!"
"What a fantastic experience! The movie kept me on the edge of my seat from start to finish."
"The movie was heartwarming, and the characters felt very real. I would definitely watch it again!"
"I had such a great time watching this film. The plot twists were incredible and the acting was flawless."
"A must-watch for anyone who loves adventure and drama! Highly recommended."
"The cinematography was stunning, and the soundtrack really enhanced the overall experience."
"Such an inspiring movie! It really made me think about life in a new way."

Komentar Negatif:
"The movie was terrible. The plot was predictable, and the acting was flat."
"I didn't enjoy this film at all. It was boring and lacked depth."
"What a waste of time! The movie was too slow, and the characters were not relatable."
"The CGI was bad, and the storyline was hard to follow. Definitely not worth watching."
"I was really disappointed with this movie. It was full of clichés and failed to engage me."
"The pacing was awful, and I found myself checking my watch multiple times."
"A dull and forgettable experience. I wouldn’t recommend it to anyone."
"The movie felt like a complete flop. The dialogue was cringe-worthy, and the plot was too predictable."


2. Click the button after filling the review.

3. See prediction and similarity search result: 

- Positive result output example:
![result-poz](/docs/negreview.png)


- Negative Result output example:
![result-neg](/docs/pozreview.png)


# How to run
Here's how you can run the app locally:

1. clone this respository 
```
git clone https://github.com/brokamal/nolimit-ds-test-reyhan-nandita-al-zahra
```
2. install dependencies
```
pip install -r requirements.txt
```
3. run app.py
```
streamlit run app.py
```
4. go to the address
```
http://localhost:8501
```

# Flowchart 
The flowchart is divided into two parts: training and inference.Training flowchart contains theprocesses of training/fine-tuning the pre-trained model distillBERT. The Inference flowchart shows the processes of running the model through inference, in this case it's deployed on Hugging Face Space. 
## Training Flowchart
![train-flow](/docs/train.png)
### Explanation 
1. Rotten tomatoes review sentiment dataset is loaded as the dataset
2. Dataset is then preprocessed (tokenazation and padding)
3. distillBERT model is trained using the dataset 
4. Fine-tuned model is evaluated using evaluation metrics. 
5. Re-train if the accuracy is low.
6. Save model if the accuracy is high, as the final fine-tuned model.



## Inference Flowchart
![inf-flow](/docs/inference.png)
### Explanation
1. User input review in form of text.
2. User input is tokenized before being fed into the fine-tuned model.
3. Model receive the input, model perform two tasks: classification and embedding extraction.
4. Classification :  Logits->argmax-> sentiment label (Positive / Negative).
5. Embedding extraction : Hidden state of [CLS] token is taken as the sentence embedding.
6. Similarity search using KNN : The query embedding compared with training set embedding using KNN. Retrives the 5 closest distance of review.
7. Final output for classification is label: positive or Negative
8. Final output for similarity search is the top 5 closest distance review with the distance itself.
de



