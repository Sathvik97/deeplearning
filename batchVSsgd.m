---------->>>>Batch Gradient descent

Batch Gradient Descent computes the gradient of the cost function using the entire training dataset in each iteration. 
This ensures precise gradient estimates but can be computationally expensive for large datasets.

---------->>>>>Stochastic Gradient Descent(SGD)

SGD updates the model parameters using the gradient computed from a single randomly selected data point in each iteration. 
This makes it faster and less memory-intensive compared to Batch Gradient Descent

----------->>>>Mini-Batch Gradient Descent

Mini-Batch Gradient Descent combines the benefits of Batch and Stochastic Gradient Descent by computing the gradient using small subsets (batches) of the dataset. 
This strikes a balance between computational efficiency and gradient accuracy.
