# Cart-ML

A decision tree classifier written from scratch in Python, based on the CART (Classification and Regression Tree) machine learning algorithm.

## Example Usage
The `driver.py` file gives an example of how to use the model, by applying it to an example dataset of Owl measurements. Visualising the model trained on this dataset results in an output of:
```
Is attribute 3 >= 3.0:
└────Y: Is attribute 4 >= 1.8:
        └────Y: Is attribute 3 >= 4.9:
                └────Y: <SnowyOwl:100.0>
                └────N: Is attribute 1 >= 3.2:
                        └────Y: <BarnOwl:100.0>
                        └────N: <SnowyOwl:100.0>
        └────N: Is attribute 3 >= 5.0:
                └────Y: Is attribute 4 >= 1.6:
                        └────Y: Is attribute 2 >= 7.2:
                                └────Y: <SnowyOwl:100.0>
                                └────N: <BarnOwl:100.0>
                        └────N: <SnowyOwl:100.0>
                └────N: Is attribute 2 >= 5.0:
                        └────Y: <BarnOwl:100.0>
                        └────N: <SnowyOwl:100.0>
└────N: <LongEaredOwl:100.0>
```
Each question (Node) has 2 answers (Children) leading from it. These can then be further questions or child nodes, which list the `<class:certainty>` pairs.

Testing the model using 'Cross Validation' results in scores like:
```
Round 1, Accuracy is 0.9556
Round 2, Accuracy is 0.9778
Round 3, Accuracy is 0.9111
Round 4, Accuracy is 0.9333
Round 5, Accuracy is 0.9333
Round 6, Accuracy is 0.9333
Round 7, Accuracy is 0.9778
Round 8, Accuracy is 0.9333
Round 9, Accuracy is 0.9778
Round 10, Accuracy is 0.9556

CART classifier has an accuracy of 94.89%, +/- 4.71%
```
These change slightly every time the Cross Validation is run, due to randomly allocating each instance to a fold (subset) as well as the probability based classification used by the algorithm.

# Other datasets
The implementation works for both numerical and text data. It expects training data to be in a table format, with each row being an instance, each column being an attribute and the final column being the class.
