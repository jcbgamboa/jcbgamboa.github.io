---
use_math: true
layout: post
title: A (Very Simple) Introduction to Representation Learning
---

(post under construction :-) )

This blog post is the result of a conversation I had with some
friends some time ago. The discussion started when a idea was raised:
that the hidden layers of a Neural Network should be called its
"memory". To say the truth, one could think that way, if he wants to
think that the network is storing in a "memory" what it has learnt.
Still, the way people tend to take it is that these are "latent
variables" that the network learnt to extract from the noisy signal
that is given to it as input.

This raised the topic of Representation Learning, which I thought I'd
discuss a little here. I would like to focus on the task of
classification, where a given input must be
assigned a certain label $y$. Let's even simplify things and say that
we have a binary classification task, where the label $y$ can be
either $0$ or $1$.
I'd like to think that I have a dataset
$\textbf{x} = \{x_1, x_2, x_3, ... \}$ composed by many inputs $x_i$.

Let's imagine what happens when we start
stacking several layers after one another. Even better, let's see
it:

![Neural Network with 3 layers](/public/...)

If we call the output of the network $y_{prediction}$,
we could represent the same network with the following formula:

$$
y_{prediction} = \sigma(W_3 \times \sigma(W_2 \times \sigma(W_1 \times x + b_1) + b_2) + b_3)
$$

(I like a lot to look at these formulas. They demystify a lot all the
complexity that Neural Networks seem to be built upon.)

As you can see (and as very well discussed in
[this great Christopher Olah's blog post](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)),
what these networks are doing is basically

 * Linearly transforming the input space into some other space
 * Non-linearly transforming the input space through the application
	of the sigmoid function.

Each time these two steps are applied, the input values are more
distinctly separated into two groups: those where $y = 0$, and
those where $y = 1$. There is, for most $x_i$ in class $y=0$ and
$x_j$ is in class $y=1$, the values in
$\sigma(W_1 \times x_i + b_1)$ and
$\sigma(W_1 \times x_j + b_1)$ will probably be better separable
than the raw $x_i$s and $x_j$s. (here, I am using the expression "better
separable" very loosely. I hope you get the idea: the values
will not necessarily be "farther" from each other, but it will
probably be easier to trace a line dividing all elements of the
two classes.)

This way, if I treat the inputs as signals, then
the input to the next layer could be thought as a _cleaned version_ of
the signal of the previous layer. By _cleaned version_ I mean
that the output of the previous (lower) layer are
"latent variables" extracted from the (potentially) noisy signal
used as input.

To make things clearer, I would like to present an example. Imagine
I gave you lots of black and white images with
digits written by hand.

![MNIST digits](/public/...)

(to keep the binary classification task, let's say
I want to divide them into "smaller than 5" and "not smaller than 5".)

The first hidden layer would receive the raw images and somehow
process them into some (very abstract, hard to understand)
activations. If you think well,
I could take the entire dataset, pass through the first layer,
and generate a new dataset that is the result of applying the
first layer to all your images:

$$
x_i^{transformed} = \sigma(W_1 \times x_i + b_1) \forall x_i \in \textbf{x}
$$

After transforming my dataset, I could simply cut the first layer
of my network:

![Neural Network with 2 layers](/public/...)

Basically what I have now is exactly the same as I had before: all
my input data $\textbf{x}$ was transformed into a new dataset
$x_i^{transformed}$ by going through the first layer of my network.
I could even forget that my dataset one day were those images
and imagine that the dataset for my classification task is actually
$x_i^{transformed}$.

Well, since we are here, what prevents me from repeating this
procedure again and again? As we keep doing this multiple times,
we would see that the new datasets that we are generating divide
the space better and better for your classification problem.

Now, there are many ways in which I can say this, so I'll say it in
all ways I can think of.

 * Each new dataset is composed by "latent variables" extracted from
	the preceding dataset.
 * Each new dataset is composed by "features" extracted from the
	preceding dataset.
 * Each new dataset is a new "representation" extracted from the
	preceding dataset.

Work on learning new representations from the data is interesting
because very often some representations extracted from the raw data
when performing a certain task may be useful for performing several
other tasks. For example, features extracted when performing image
classification may be "reused" for performing, say, Visual Question
Answering (where a model has to answer question about an image).


### However

There is a catch on what I said.

I spent the post saying that, at each step, the layers would separate
the data space better and better for the task we are performing.
If that is the case, then any network with A LOT of layers would
perform very good.

But it turns out that only in ~2006 people started managing to use several layers (up to then, many believed that more layers only disturbed the training, instead of helping). Why? The problem is that these same weights that may help in separating the space into a better representation, if badly trained, may end up transforming the input into complete noise. If the "entropy" (I am using the word loosely, but the word is actually the right one) of the next representation is so high that the "structures" that were present in the previous layer are transformed into noise, then recovering the information in the subsequent layers may be impossible. If you want to try an example, make a network with several "Dense" layers initialized with completely random weights (none of these fancy "Xavier initialization" nor Batch Normalization, nor Dropout, nor anything) and try naïvely training it. It is possible that it will simply "diverge" and not learn anything. (True, however, I am  ignoring other points that may be causing the network to fail)


Conclusion
----------

These ideas are very powerful and I hope the intuitions here make
them easier to use and to extend.

### 


