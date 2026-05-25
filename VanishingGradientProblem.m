-------------------------------------------------VANISHING GRADIENT PROBLEM----------------------------------------------------------------------------------------
The vanishing gradient problem refers to the issue of diminishing gradients during the training of deep neural networks.
It occurs when the gradients propagated backward through the layers become very small, making it difficult for the network to update the weights effectively.
The vanishing gradient problem can hinder the training of deep neural networks.
It slows down the learning process, leads to poor convergence, and prevents the network from effectively capturing complex patterns in the data. 
The network may struggle to update the early layers, limiting its ability to learn meaningful representations.

-------------------------------------------HOW TO REDUCE VANISHING GRADIENT PROBLEM-------------------------------------------------------------------------------

1) REDUCE THE COMPLEXITTY--> reduce the complexity of the neural network maybe decrease the number of hidden layers,but this method is not that reliable to do so
not used much though

2) USING RELU ACTIVATION FUNCTION--> f(x)=Relu=max(0,z),when the variable enters relu it returns the max of the variable and 0

3) Proper Weight Initializations--> We can solve the issue by actually initializing the weights properly

4) Using Batch Normalization--> Its a layer 

5) Using a residual network
---------------------------------------EXPLODING GRADIENT PROBLEM---------------------------------------------------------------------------------------------------

Basically the partial derivative is very big number happens in recurrent neural networks a lot and when we update the weight the loss keeps increasing and the model
starts behaving randomly cuz the loss is randomly oscilating
