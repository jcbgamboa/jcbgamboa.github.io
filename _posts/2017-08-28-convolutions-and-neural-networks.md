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

![Lenna original](/public/lenna.bmp)

The first thing to do is to load the image:

```python
from PIL import Image
img = Image.open('lena.bmp')
```

Then I want to create a function to convolve the image
with the kernel:

```python
# import numpy as np

def convolve(image, kernel):
	# Flips the kernel both left-to-right and up-to-down
	kernel = np.fliplr(np.flipud(kernel))

	# Transforms the image into something that numpy can process
	image_array = np.array(image)

	# Initializes the image I want to return
	new_image_array = np.zeros(image_array.shape)

	# Convolve
	for i in range(image_array.shape[0] - kernel.shape[0]):
		for j in range(image_array.shape[1] - kernel.shape[1]):
			# run_kernel will perform the pointwise multiplication
			# followed by sum
			new_image_array[i][j] = run_kernel(image_array, kernel, i, j)

	# Creates a new Image object
	new_image = Image.fromarray(new_image_array)

	# Returns both the image as an array, and as an Image object
	return new_image_array, new_image
```

As you can see, I am using `numpy` to perform the calculations. I
expect you not to find it hard to understand the code. It could
obviously be written much more efficiently (`numpy` actually even
has a function that performs the convolution anyway), but I wanted
to show how the operations we saw in the last blog post can be easily
translated into some piece of code.

Now we need to define that `run_kernel()` function. It calculates
$\odot$ operation between the part of the image that we are interested
in and the (already flipped) kernel. This is as simple as:

```python
def run_kernel(image, kernel, pos_x, pos_y):
	ret = 0
	for i in range(kernel.shape[0]):
		for j in range(kernel.shape[1]):
			ret += image[pos_x + i][pos_y + j] * kernel[i][j]

	return ret
```

Done! It is that simple!

What we are missing is just the right kernel. If you look at the
Wikipedia page you'll see that there are several kernels usable for
Edge detection. I'll use the third one:

$$
kernel =
\begin{bmatrix}
-1 & -1 & -1 \\
-1 &  8 & -1 \\
-1 & -1 & -1
\end{bmatrix}
$$

In Python:

```python
new_image_array, new_image = convolve(img, np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]))
new_image.show()
```

With this, you should see the following image:

![Lenna after edge detection](/public/lenna_edges.bmp)

Nice, right?

### The Border Problem

If you look carefully at this new image, you'll see that I'm not
running `run_kernel()` in the last pixels (and then you'll find some
columns of zero pixels at the right of the image, as well as some
some rows at the bottom). This has to do with what I called the "Border
Problem" in my last post.

It is actually very unclear what should be done in the edges of the
Image we are trying to process. The way I have been doing so far, if I
calculate a convolution between two $3 \times 3$ matrices, it will
give me only one number. It would be nice if I could find ways to get
a result that had the same size of the input image.

For this reason, you will see three types of convolutions:

 * **Valid**: This is the way I have been doing it so far. We don't
	assume any information apart from what we have.

 * **Full**: In this case, we assume there are lots of zeros beyond
	that the edge of the original image. This way, if we were
	given the image $f$ below, then it would be "transformed" into
	the $f_{transformed}$ below before convolving. The number of
	new rows/columns introduced depends on the size of the kernel.
	This makes sense from the perspective of signal processing I
	described in my previous post.
	_(if this is not clear enough, you are welcome to take a look at
[this amazing explanation I found in Stack Overflow](https://stackoverflow.com/a/37146742/1360979))_

$$
f = 
\begin{bmatrix}
0 & 3 & 6 & 3 \\
3 & 6 & 3 & 6 \\
6 & 3 & 6 & 3 \\
3 & 6 & 3 & 0 \\
\end{bmatrix}
$$

$$
f_{transformed} =
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 0 & 0 \\
0 & 0 & 0 & 3 & 6 & 3 0 & 0 \\
0 & 0 & 3 & 6 & 3 & 6 0 & 0 \\
0 & 0 & 6 & 3 & 6 & 3 0 & 0 \\
0 & 0 & 3 & 6 & 3 & 0 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 0 & 0 \\
\end{bmatrix}
$$


 * **Same**: This is a little trickier. It does assume zeros around
	the image, but only as much as needed to return an output that
	has the exact same size as the input image. I tend to find it
	hard to visualize, but I found that
	[this image](http://www.johnloomis.org/ece563/notes/filter/conv/convolution.html)
	helped a lot.


Relation to Convolutional Neural Networks
-----------------------------------------




Bonus: Shifting a Signal
------------------------

Neural Turing Machines...



Conclusion
----------


