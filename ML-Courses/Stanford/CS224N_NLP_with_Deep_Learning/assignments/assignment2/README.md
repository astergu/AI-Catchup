### (a) (2 points) 

> Prove that the native-softmax loss is the same as the cross-entropy loss between y and $\hat{y}$, i.e.  $-\sum_{w \in Vocab} y_w log(\hat{y}_w) = - log(\hat{y}_o)$

> Answer: <font color='red'> $y_w$ is one-hot vector with a 1 for the true outside word o, and 0 everywhere else. Thus all the sum terms whenever w $\neq$ o are zeros. </font>

### (b) (7 points)
> (i) Compute the partial derivative of $J_{naive-softmax}(v_c, o, U)$ with rescpect to $v_c$. Please write your answer in terms of $y$, $\hat{y}$, $U$, and show your work to receive full credit.

> Answer: <font color='red'> The partial derivative of $J_{naive-softmax}(v_c, o, U)$ with respect to $v_c$:  
> $\pdv{J_{naive-softmax}(v_c, o, U)}{v_c}$
> </font>


> (ii) When is the gradient you computed equal to zero?

> Answer: <font color='red'> </font>


> (iii) The gradient you found is the difference between two terms. Provide an interpretation of how each of these term improves the word vector when this gradient is subtracted from the word vector $v_c$.

> Answer: <font color='red'> </font>