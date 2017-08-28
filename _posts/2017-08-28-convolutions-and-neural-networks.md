---
use_math: true
layout: post
title: Convolutions and Neural Networks
---

(post under construction :-) )

In my last blog post, I took you by the hand and guided you through
the realm of convolutions. I hope to have made it clear why it makes
sense to discretize functions and represent them as vector, and how
to calculate the convolution of 1D and 2D vectors.

In this post I want to talk a little about how Image Processing was
done in the old times, and show the relation between the procedures
performed back then and the kinds of parameters learnt by
Convolutional Neural Networks (CNN). In fact, do notice that CNNs
have been lurking around for years
([LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
had been introduced in 1998!) before they went viral again in
2012 (with the AlexNet), so, in a way, they are concurrent models to
the models described below.

It is hard to tell why Convolutional Neural Networks took so long to
become popular. One reason might be that Neural Networks
had gone somewhat out of fashion for a while until their revival
some years ago.
([Hugo Larochelle](the fact://www.youtube.com/watch?v=dz_jeuWx3j0)
commented in this TEDx video how there were papers that were rejected
simply based on the argument that his approach used Neural Networks.)

Another contributing factor might be that, for a long time, it was a
common belief for many people that Neural Networks with many layers
were not good (despite the work with
[LSTMs](https://en.wikipedia.org/wiki/Long_short-term_memory) being
done in Europe). They were taken as "hard to train" and empirically
many experiments ended up producing better performances for models
with just a few (or even only one) layer. CNNs, however, did not
suffer from these problems (at least not that much), and the LeNet
paper from 1998 had already 5 layers.

But my focus here is not on the architecture of CNNs, nor on their
gradient flow or their history. My focus here is on how exactly we
can say that the shared weights of a CNN results in a mathematical
formulation that is identical to that of the Convolutions that we
discussed in the previous post.


Image Processing 
----------------

Before I go into the CNNs I want to show why a Convolutional is
something that we might want to do to an image. In my previous post,
I tried to be as generic as possible, talking about functions and
vectors, talking about things from a "signal processing" point of
view. It turns out that the Image Processing community has its own
perspective. So, from now on, I will take $f$ as a 2D image that I
want to somehow process, and to $g$ as a
[_kernel_](https://en.wikipedia.org/wiki/Kernel_(image_processing)).

When we learn math in school, we learn names of several functions that
are known to be useful, and somehow represent well parts of the world
we live in. Examples of such functions are $log$, $ln$, $sin$, or
$tg$.
When we are introduced to statistics, we get acquainted to several
other names, such as "correlation", "standard deviation", "variance",
"mean" or "mode". The types of kernels used in Image Processing are
not different: researchers in the area have found through the years
several kernels that are known to perform well different kinds of
tasks, such as _blurring_, _edge detection_, _sharpening_, etc.
You can find a list of such kernels in the
[Wikipedia article](https://en.wikipedia.org/wiki/Kernel_(image_processing).

I want to show how a convolution could be used to find the edges
of an image. But this time, I don't want to show formulas; I think
some Python code should make things clearer. Let's say we want to
find the borders of the following image of Lenna
[Lena](https://en.wikipedia.org/wiki/Lenna):

![Convolution of a function with itself.](/public/lenna.bmp)

The first thing to do is to create a function to convolve the image
with the kernel:

```
def convolve(image, kernel):
	image_array = np.array(image)
	new_image_array = np.zeros(image_array.shape)
	for i in range(image_array.shape[0] - kernel.shape[0]):
		for j in range(image_array.shape[1] - kernel.shape[1]):
			new_image_array[i][j] = run_mask(image_array, kernel, i, j)

	new_image = Image.fromarray(new_image_array)
	return new_image_array, new_image
```


### The Border Problem





Relation to Convolutional Neural Networks
-----------------------------------------



Bonus: Shifting a Signal
------------------------

Neural Turing Machines...



Conclusion
----------

