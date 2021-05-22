# ReadMe: Subreddit Classification of Fact and FakeFacts

## Problem Statement
The purpose of this project was to create a model that can reasonably delineate between subreddits that contain facts, and fake facts respectively, for the purpose of being used as a feature in a seletive solical media filter. Though our sources are both well moderated , this model does NOT discreminate between actual facts and fake facts, but rather between which media outlet a post originated, expanding on this model with a dataset of verified facts and fake facts would be a step for future projects.

## Sources 
I used the pushift API to gather 2,500 post from 2 seprate subreddits. The two sources are reffered to throughout the code for this project as follows.

fact
https://www.reddit.com/r/facts/

fiction
https://www.reddit.com/r/FakeFacts/

A brief example of observations from each source. We independandly verified the factual/ non factual nature of the posts of 25 independant authors. Each of them were found to be accurate.  

|c/FakeFacts|r/facts|
|---|---|---|
|(Marine Fact: Lions Mane Jellyfish have triple helix shaped DNA. |(Libraries in this world contain alot of Books bound in Human flesh.) |
| (Fortnite was created by the Catholic church to stop people having sex before marriage.)|(Only female mosquitoes bite. They need the blood to reproduce. Male mosquitoes eat flower nectar.) |
|(2018 marks the 97th anniversary for Albert Einstein winning the Nobel prize, for his now-famous Theory of Relativity.)| (You can’t flick things while smiling, don’t trust me, then try it) |



# Executive Summary 

## Model Selection
___
We ran testing on an extensive number of models. Here is a table of the models that wer kept for comparison. 


|model|Test|Train|Gap|
|---|---|---|---|
|Support Vector Machines TFIDF| 0.990587 | 	0.776973 |	0.213613 |
|Multinomial Naive Bayes TFIDF |0.937002 |	0.756698 |	0.180304 |
|Logistic Regression| 0.996379 | 0.765387 |	0.230992 |


## Exlploring the Lexicon

Something we noticed right away which was outside of the lexicon was that 30% of the factual posts have links to sources, while only 5 % of the FakeFacts posts had similar links. We have decided to engineer a feature for Has_link to inlcude in our final filter model. Assessing the type of link is a step to be explored in a latter phase.

Our analysis of posts showed their to be an standard deviation of character length shared between both sources. Post from the r/Facts subreddit tended to be only slightly longer than those in its counterpart. 

Once we had cleaned punctuation and miscelaneous improperly formatted text from our dataset we began to look at the overlap in most frequently used words in the lexicons of both subreddits. Our goal here was to isolate the most common words shared between both Datasets and remove them from the corpus entirely so as to reduce some dimensionality. This drammatically improved reduced the gap between our Train and Test Accuracy Scores, but did little to imporve ouru overall performance.

After our iterative removal of ovelapping words we were left with these two uniques lists of common words from both sources. They are listed below in decending order or frquency.

| Fact | Fiction |
|---|---|
| amp | called |
| source | used |
| org | named |
| medium | word |
| wiki | originally |
| life | make |
| html | invented |
| youtube | new |
| human | real |
| 2020 | way |
| news | use |
| history | known |

We are able to gain some valuable insight from this. It looks like a factual post is likely to contain a reference to a source, eiter wikipedia, medium or youtube most commonly. At the same time we can get a peak into the kinds of misinformation being spread. Statements about new things, who originally did whaT , who invented what, or what is real.


## Conclusion 

With relatively litte cleaning or reorganization of text scikit learns NLP toolkit porved effective at imedeitely iproving our accuracy by 25 % over baseline. As it stands their are 3 main paths forward of this project.

1) collect more training data 
2) apply adaboost classification 
3) incorporate voter classification 
4) incorporate output into media filter