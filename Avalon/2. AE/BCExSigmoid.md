**the combination with loss using BCE and Sigmoid as the activation function result in a beautifull equation**

lets make sure we are on the same page, so here are the variables
- Logit ($z$): Input to activation function
- Sigmoid Activation: ($\hat{x}$): $\hat{x} = \sigma(z) = \frac{1}{1 + e^{-z}}$
- Loss ($L$): $L = -[x \log(\hat{x}) + (1-x) \log(1-\hat{x})]$


because we have 2 main components:
1. first part = $x \log(\hat{x})$
2. second part = $(1-x) \log(1-\hat{x})$

now the derivative process, remember derivative from $\log(u)$ is $\frac{1}{u} \cdot \frac{du}{d\hat{x}}$

so, derivative for first part:

$$\frac{d}{d\hat{x}} [x \log(\hat{x})] = x \cdot \frac{1}{\hat{x}} = \frac{x}{\hat{x}}$$

and, for second part we gonna have to use chain rule, we focus to derivate the $\log(1-\hat{x})$ part, hence the derivate of $\log$ is $\frac{1}{1-\hat{x}}$  then we multiply with the derivative inside it (the $1-\hat{x}$, we derivate into -1). so at the end we have the result of $-\frac{1}{1-\hat{x}}$. dont forget with the $(1-x)$ part. so we multiply the both side $(1-x) \cdot \frac{-1}{1-\hat{x}}$ 

and so we got the final result for second part:

$$-\frac{1-x}{1-\hat{x}}$$

Now we have both in form after derivative, then we have to combine it, 

so from

$$-[x \log(\hat{x}) + (1-x) \log(1-\hat{x})]$$

transform into

$$\frac{dL}{d\hat{x}} = -\left[ \frac{x}{\hat{x}} - \frac{1-x}{1-\hat{x}} \right]$$

we can continue it to do simplification, we can make the dominator to be equal 

$$\frac{dL}{d\hat{x}} = -\left[ \frac{x(1-\hat{x}) - \hat{x}(1-x)}{\hat{x}(1-x)} \right]$$

after multiplying the outsider with insider (hehe i dont know how to say it)

$$\frac{dL}{d\hat{x}} = -\left[ \frac{x - x\hat{x} - \hat{x} + x\hat{x}}{\hat{x}(1-\hat{x})} \right]$$

we got 

$$\frac{dL}{d\hat{x}} = -\left[ \frac{x - \hat{x}}{\hat{x}(1-\hat{x})} \right]$$

the negatif ("-") we multiply with the numerator result in 

$$\frac{dL}{d\hat{x}} = \frac{\hat{x} - x}{\hat{x}(1-\hat{x})}$$

now we want to look for derivative of L with respect of z ($\frac{dL}{dz}$). because we want to see what would happen if we change the z, and for this we have to use chain rule:

$$\frac{dL}{dz} = \frac{dL}{d\hat{x}} \times \frac{d\hat{x}}{dz}$$

because to get $z$ to be responsible to the loss $L$ first we have to create a bridge with $\hat{x}$ because reconstruction image is create the lose (so its responsible to the loss) and it also created by the input $z$ (from the neuron, so the input are responsible to it) 

and remember we already computed the first part $\frac{dL}{d\hat{x}}$ so now all we have to do is just doing the derivative for sigmoid to the respect of $z$ ($\frac{d\hat{x}}{dz}$), and what so special about sigmoid is, the derivative can be written as the function it self.

$$\frac{d\hat{x}}{dz} = \hat{x}(1 - \hat{x})$$

now if we multiply both of this derivative

$$\frac{dL}{dz} = \underbrace{\left( \frac{\hat{x} - x}{\hat{x}(1 - \hat{x})} \right)}_{\frac{dL}{d\hat{x}}} \times \underbrace{\hat{x}(1 - \hat{x})}_{\frac{d\hat{x}}{dz}}$$

and so we got the final result of 

$$\frac{dL}{dz} = \hat{x} - x$$

meaning the derivative of loss with respect of the