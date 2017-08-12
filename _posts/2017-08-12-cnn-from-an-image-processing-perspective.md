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
What curve exactly? I'll discuss more about it in the next section.
For now, what is worth is to understand that there several ways in
which you can think of convolutions, and it might help a lot if
you allow yourself to switch views at different points in time.


A concrete example
------------------

If you go to the
[Wikipedia article on convolutions](https://en.wikipedia.org/wiki/Convolution),
you may find the following two (awesome) images:

![Convolution of a function with itself.](public/convolution.gif)

![Convolution of a spiky function with a box.](public/convolution2.gif)


What these images are saying is that you can calculate the value of the
convolution $f \ast g$ at the point $t$ by following a very simple
procedure. I'll define two functions $f$ and $g$ to make the steps
easier to follow. Let

$$
f(x) =
\begin{cases}
  1 & \text{if } 0 \leq x \leq 1 \\
  0 & \text{otherwise}
\end{cases}
$$

and

$$
  g(x) = 2 * f(x)
$$

Here we have the two curves:

![Two signals](public/grid1.gif)


**First**: flip $g$ horizontally at the point $t$.
Let's give the flipped $g$ a name, say $g'$. (if you don't flip $g$,
then what you are calculating has actually the name of "correlation",
and is simply another typical operation in signal processing.)

![Flipped signal](public/grid2.gif)


**Second**: shift $g'$ horizontally by $t$ units. If $t$ is
positive, then $g'$ will be shifted to the right; otherwise, it will
be shifted to the left.

![Shifted signal](public/grid3.gif)

**Third**: this is the step where the problems arise.
Now what you want is actually multiply the two
curves are each point between $-\infty$ and $+\infty$ and calculate the
area below the curve that this multiplication will form. But there are
two points that could make your life easier. The first of them is that
it is worth noting that very often
the functions $f$ and $g$ for which we want to calculate a
convolution are 0 in most of their domain.
Therefore it may be the case now that the non-zero parts of $f$ and
the shifted version of $g'$ do not even intersect, or only intersect
in some places (by "intersection", I mean that $g'$ and $f$ are non-zero
for the same $x$ inputs). Let's say they intersect in an
interval $[a, b]$. Now it could still be a challenge to calculate the
integral of the shifted $g'$ and "f" in that interval.

![Calculate area below curve](public/grid4.gif)

(While searching for a way to understand this procedure, I came across
[this very nice demo](http://www.fit.vutbr.cz/study/courses/ISS/public/demos/conv/).
In it you can define your own functions and play arround to find out
how the convolution is going to be.)

Now, the problem with
continuous convolutions is that we would have to actually calculate
an integral. But what if our function were actually "discrete"?

![Calculate sum of elements below curve](public/grid5.gif)

All the concepts we have discussed so far would follow the same logic
for the discrete case. Now,
instead of an integral we now have a sum. So, given the interval
$[a, b]$, we could calculate the convolution as

$$
  \sum^b_a{f(i) * g(j)}
$$


This is precisely the kind of operation we need for image processing...


2D discrete convolutions
------------------------




