---------------------------------------------DROPOUT-------------------------------------------------------------------------------------------------------------


Overfitting is a common challenge in deep learning, where a neural network performs well on training data but struggles with new, unseen data. 
One effective technique to combat this is Dropout—a regularization method that prevents overfitting by randomly “dropping” nodes in a neural network during training.


----->HOW DROPOUTS WORK-->
1) larger and more complex neural networks might lead to overfitting hence in dropout we drop nodes randomly in the training iterations to form smaller neural 
networks ie less complex and see what happens 
2) during the testing face however all the neurons are ACTIVE but the weights are scaled down to the value of weights x (1-p) where p is a hyperparameter and 
represents the probability or fraction of drop during the training iterations


----->WHY DROPOUT WORKS--->
By randomly disabling neurons, dropout effectively trains multiple “subnetworks” within the main network, which makes the overall model more adaptable and prevents it
from memorizing specific data points (overfitting). 
This technique has shown to improve model accuracy by up to 2%, especially on complex datasets.



