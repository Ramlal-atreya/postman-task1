---INTRO

The given task was to make a feed forward neural network with manual backpropagation
and to compare my results with libraries like torch
also to make a plot to show that the loss factor is decreasing with more training


---WORKING


The neural network is a fully feed forward neural network with one hidden layer


---PROCESS


I initially started off by watching the 3blue1browm video and understood the basic fundamentals and math behind neural networks

Then I fed the  Neural Networks and Deep Learning, Chapter 2 — Michael Nielsen to an LLM and asked it to teach me everything

I then read the attached pdf in the prior resources and learn about bias vs variance and other statistical stuff.

By using the basic python I already knew I started learnt numpy and started with a random youtube video on how to code basic nueral networks


--After I was done, I set up GitHub and made my initial commit.


By using the knowledge I acquired from the youtube video I made the basic framework i.e the neuron_forward function.

I defined the matrixes Z1, A1, Z2, A2. It took me alot of time to figure out what a sigmoid function was and why it was being used😭😭

sigmoid is a function that changes any value into the range of (0,1).

the sigmoid I used is y = 1/1-e^-x


-- This is when I made my first commit.


Added the sigmoid derivative function and spent a few minutes trying to optimize it


-- Second commit


Applying the math that I learnt in the videos that I watched and the chapter that I read I started writing the gradient equations.

We know that z = w*a + b

where w = weight a = the previous activation and b = bias.

and a = sigmoid(z)

and the loss function is  C(loss) = (y-a)^2

y is the expected output during training and a is the final prediction or output.

So from the above equations, we know that there are 3 factors effecting the loss

~weight ~previous activation ~bias

and our goal is to reduce the loss function to the minimum value

We do this by finding out the gradient which is nothing but the slope

Take a partial derivative of the loss function with respect to the weight, previous activation(lets call it u for now as im using a for the final output in the loss function), and bias

I will also be using ∂ as the partial derivative symbol

now ∂C/∂w is gradient with respect to weight

∂C/∂u is gradient with respect to the previous activation

∂C/∂b is gradient with respect to the bias

Generalizing, lets say we have ∂C/∂x where x is a random variable for now we can write this as

∂C/∂a * ∂a/∂z * ∂z/∂x   ~~~chain rule of derivatives

so  ∂C/∂a * ∂a/∂z is common for all lets derive this first

since c = (y-a)^2  ∂C/∂a is 2*(a-y)  and ∂a/∂z is just the derivative of the sigmoid function  sigmoid`(z) {im using ` to imply derivative}

Therefore, ∂C/∂a * ∂a/∂z = 2*(a-y)*sigmoid`(z). Lets call this K for now for convenience.

1) with respect to weight

	∂C/∂w = K * ∂z/∂w  and since z = w * u + b ∂z/∂w is u
	∂C/∂w = k * u

2) with respect to previous activation

	∂C/∂u = K * ∂z/∂u  and since z = w * u + b ∂z/∂w is w
	∂C/∂w = k * w

3) with respect to the bias

	∂C/∂u = K * ∂z/∂b and since z = w * u + b ∂z/∂w is 1
	∂C/∂w = K * 1 = K

Therefore, the gradient matrixes

ΔW, ΔU and ΔB will be made and we adjust the actual W,  and B matrixes using these values based on our lr

I used lr = 0.1.


-- Third commit


fixed a logic error

I initialized these ΔW, ΔU and ΔB matrixes inside the forward function instead of inside the actual code so when I realized this I brought them outside the function.


-- Fourth commit


I thought I was done but then I realized I wasn't printing anything and therefore no output

so I used the .shapes() function to print the numpy arrays


-- Fifth commit


Now came the time to test my NUERAL NETWORK OMG LES GOOOO!!!!

I gave the input matrixes X and Y simply and it gave a bunch of errors and everything 

Then I realized that I wasn't using numpy arrays

I fixed that error as well and put in inputs and it worked and it was the expected output.


-- Sixth commit


I then made the comparison tester by using torch

To be honest I didn't learn much of torch I just copy pasted the commands

made the comparison tester using the np.allclose() function.


-- Seventh commit


I started with the training program 

watched a youtube video to figure out how to use MNIST

changed the input hidden and output order to 764 64 and 10

it wasn't really a big process because i made almost everything required for it already. I copy pasted most of the required things from my previous code the only new thing was the training for loop

that is the actual adjustments to the W,U and B matrixes.

I added the SoftMax function which was pretty interesting and also the cross_entroy function.

the last thing i did was the matplotlib.pyplot. I tried learning it but it was too complex for me so i just used AI to add the plots

The test went well and we could clearly see that the loss was minimizing.


-- Eight commit


I tweaked with the lr value a bit and ended up with 0.1 again because i felt it was best for the 2000 training instances that i used.

and also added the seed for the numpy random rand function.


-- Ninth commit


Wrote the Writeup.mb (4th wall broken haha).

-- Thenth commit

started with the stretch using ReLU and MSE

coded it simple and made the comparison

-- Eleventh commit

The graph was not normalized so botht he scripts ware outputting on different scales
so i got them into the common scale

-- Twelth commit

I tweaked the lr value so that both were represented well
any lr over 0.35 caused a lot of noise in the ReLU and MSE graph 
but the original needed an lr of 0.6 to reduce loss quickly
so took the best of both worlds at 0.4.

Only thing left is writing the Readme and IM DONEEE!!!!












