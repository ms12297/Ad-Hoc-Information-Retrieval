# Ad Hoc Information Retrieval System

## Overview

In this project, I match queries with abstracts in the cran collection of documents based on the TF-IDF and Cosine Similarity scores of the relevant vectors. These vectors are constituted by all the unique words encountered as features. 

## Running the System:

Before running the system, make sure to place cran.qry and cran.all.1400 in the same folder as the script adhoc_ir_system.py.
To run the system, run the following command in your terminal:
python3 adhoc_ir_system.py
The system will generate an outfile file named output.txt which contains the top 100 ranked (highest cosine similarity to lowest) abstracts for each query.

## Implementation:

Vectors for all the abstracts and the queries have the same features, with the same length representing the same set of words. I use a set of words to collect the unique words encountered in both all abstracts or all queries for setting features. For this reason, I also implement a smoothing of +1 in all log formulas to account for the case of the undefined log(0) that occurs with zero counts. 

I use the standard split() function for tokenization. Furthermore, I use the nltk library to strip the input of all puntuation and I use PorterStemmer to incorporate stemming.
