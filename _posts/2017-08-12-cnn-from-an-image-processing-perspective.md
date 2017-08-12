---
use_math: true
layout: post
title: CNNs from an Image Processing Persepective
---

(post still under construction :-P )

For quite some time already I have been wanting to write this blog
post. A little more than one year ago I got acquainted to
Convolutional Neural Networks, and it didn't immediately strike why
they are called that way. I eventually read
[this blog post](http://colah.github.io/posts/2014-07-Understanding-Convolutions/)
that helped a lot to clarify things; but I thought I could try to
give more details on what exactly is meant when one says 
"Convolution" here.

This blog post builds upon the description given
[there](http://colah.github.io/posts/2014-07-Understanding-Convolutions/),
so, if you still didn't read that, stop reading this and go there
take a look at that blog post.


Convolutions
------------

Convolutions are a very common operation in signal processing. While
the [colah's blog post](http://colah.github.io/posts/2014-07-Understanding-Convolutions/)
presents it in a more abstract/intuitive statistical way, I find that
a more gore calculus-driven introduction from Kahn Academy might help
you realize that the concept is just an integral:

<iframe width="560" height="315" src="https://www.youtube.com/embed/IW4Reburjpc" frameborder="0" allowfullscreen></iframe>

In this
Kahn Academy video, Sal found a closed formula for the convolution
by solving the integral. Given that a convolution is an integral,
you might consider that it represents the area below some curve.
Keep that in mind =)

While finding a closed formula for the convolution may be useful,
there will be times when we may be interested
in the value of the convolution only for certain points. That is,
there are only a handful of points $t$ for which we are interested in
calculating $(f \ast g)(t)$. For these cases, it may be worth not
to use calculus (or integrals) to calculate $(f \ast g)(t)$. The
next section will calculate the convolution using a very simple
procedure that should shed some light on how it may be used for
image processing.

My point here is: there several ways in which you can think
of convolutions, and it might help a lot if you allow yourself to
switch views at different points in time.


A continuous example
--------------------

It is worth noting that very often
the functions $f$ and $g$ for which we want to calculate a
convolution are 0 in most of their domain. Let us define some $f$ and
$g$ for which the integral is easy to calculate. Say:

$$
f(x) =
\begin{cases}
1 & \text{if } 0 \leq x \leq 1
0 & \text{otherwise}
\end{cases}
$$

and $g(x) = 2 * f(x)$.

These 

In these cases, the procedure
presented in these images makes our life much easier:

![Convolution of a function with itself.](public/convolution.gif)

![Convolution of a spiky function with a box.](public/convolution2.gif)

_(These images were taken from the
[Wikipedia article on convolutions](https://en.wikipedia.org/wiki/Convolution),
which is another awesome resource on the topic, by the way.)_

What these images are saying is that you can calculate the value of the
convolution $f \ast g$ at the point $t$ by following a very simple
procedure.

**First**, flip $g$ horizontally at the point $t$.
Let's give the flipped $g$ a name, say $g'$. (if you don't flip $g$,
then what you are calculating has actually the name of "correlation",
and is simply another typical operation in signal processing.)

**Second** shift $g'$ horizontally by $t$ units. If $t$ is
positive, then $g'$ will be shifted to the right; otherwise, it will
be shifted to the left.

**Third**: it may be the case now that the non-zero parts of $f$ and
the shifted version of $g'$ are intersecting ($g'$ and $f$ are
non-zero for the same $x$ values). In that case, 

flipping $g$ horizontally (if you don't flip, then it
becomes a "correlation", another common operation in signal processing)
and calculating, for each point $t$, the area of intersection between
$f$ and a  $\hat{g}$, where $\hat{g}$ is 


Basically, if you have a function $f$ and another function $g$, then
you can calculate the convolution $f \star g$ following this
procedure:



