This is one of two repositories for my master's thesis:
1. Data scraping and processing: https://github.com/solfang/Social-Media-Data-Pipeline
2. Analyis: This repo

# Iconic Architecture on Instagram 

The code was run using Python version 3.7.
If something doesn't work for you, please create an issue.

## Setup: getting the data
1. Download the [data](https://drive.google.com/file/d/1hPFtlyjCfInvKOSjnTvPxulOCznDC10v/view?usp=share_link) and  unzip it one folder layer above this repository. The resulting folder should be:
- [...]/some folder/
	- [this repo]/
	- data/
2. Run `python -m spacy download en_core_web_sm` (required for the NLP notebooks)
3. Optional: get the images. images are not included in the above link and without them, most code in ImageAnalysis.ipynb won't run. The images can be obtained from the TUM urban development drive and are under [...]/data/images/

## The notebooks

| **Notebook**      | **Analysis**                                                  | **Method**                                                                                                                                |
|-------------------|---------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| ImageAnalysis     | Exploratory analysis                                          | Knn-clustering on Image descriptors computed with [DIR](https://github.com/naver/deep-image-retrieval)                                    |
|                   | Image label evaluation                                        | Image labels computed with [Places-365 CNN](https://github.com/CSAILVision/places365)                                                     |
|                   | Image content analysis                                        | Clustering-based Image classification                                                                                                     |
| NLP_Basic         | Mentions of the architect in hashtags                         | Hashtag frequency                                                                                                                         |
|                   | Performance of image content in terms of likes/comments       | Plot by image class                                                                                                                       |
|                   | Most common adjectives/verbs                                  | Part-of-speech tagging                                                                                                                    |
| NLP_TopicModeling | Topic modeling                                                | [BERTopic](https://maartengr.github.io/BERTopic/index.html)                                                                               |
| PostFrequency     | Post activity development over time and with relation to city | Post frequency analysis                                                                                                                   |
|                   | Seasonality                                                   | [Autocorrelation](https://medium.com/@krzysztofdrelczuk/acf-autocorrelation-function-simple-explanation-with-python-example-492484c32711) |
|                   | Events and peaks in post activity                             | Peak finding                                                                                                                              |