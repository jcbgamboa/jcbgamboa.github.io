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


**First**: flip $g$ horizontally.
Let's give the flipped $g$ a name, say $g'$. (if you don't flip $g$,
then what you are calculating has actually the name of "correlation",
and is simply another typical operation in signal processing.)

![Flipped signal](public/grid2.gif)


**Second**: shift $g'$ horizontally by $t$ units. If $t$ is
positive, then $g'$ will be shifted to the right; otherwise, it will
be shifted to the left. I'll call this function $g_{shifted}'$

![Shifted signal](public/grid3.gif)

**Third**: this is the step where the problems arise.
Now what you want is actually multiply the two
curves are each point between $-\infty$ and $+\infty$ and calculate the
area below the curve that this multiplication will form.
Let's assume that the functions are zero most of the time (just like
in our example), and non-zero only in a small section of their domain.
Because we are multiplying the two values, we only care about the values
where both functions are not 0. In all other cases, the integral will
be 0 anyway. Let's assume that both functions are non-zero only in an
interval $[a, b]$. In this case, our problem reduces to calculating the
integral of the multiplication of $f$ and $g_{shifted}'$ inside that
interval. Now it could still be a challenge to calculate the
integral of the $g_{shifted}'$ and "f" in that interval.

![Calculate area below curve](public/grid4.gif)

(While searching for a way to understand this procedure, I came across
[this very nice demo](http://www.fit.vutbr.cz/study/courses/ISS/public/demos/conv/).
In it you can define your own functions and play arround to find out
how the convolution is going to be.)

The problem with
continuous convolutions is that we would have to actually calculate
an integral. But what if our function were actually "discrete"?
Fortunately for us, most applications on Image Processing require
discrete signals, and for our purposes it would be perfectly ok to
discretize these continuous signals.

![Calculate sum of elements below curve](public/grid5.gif)

After discretization, All the concepts we have discussed so far would
follow the same logic. Now,
instead of an integral we now have a sum. So, given the interval
$[a, b]$, we could calculate the convolution as

$$
  (f \ast g)(t) = \sum^b_{i=a}{f(i) * g_{shifted}'(i)}
$$

And fortunately this sum is easy to calculate.

1D discrete convolutions
------------------------

It turns out that the functions $f$ and $g$ used in convolutions are
in reality mostly composed by zeros (as assumed before). This allows
for a much more compact representation of the functions as a vector of
values. For example, $f$ and $g$ could be represented as:

$$
f = [\dots 0, 0, 1, 1, 1, 1, 0, 0, \dots] \\
g = [\dots 0, 0, 2, 2, 2, 2, 0, 0, \dots] \\
$$
_(Of course, the number of ``1" and ``2" depends on how the discretization was performed)_

Now let's say I'd like to calculate the value of the convolution
between $f$ and $g$ at the point $t = $*some coordinate*. It is hard
to point the exact place, so I'll make the place bold:

$$
f = [\dots 0, 0, 1, 1, \textbf{1}, 1, 0, 0, \dots] \\
$$
_(For future reference, I'll call this position $t=2$)_

The way to calculate it is just the same:

 1) Flip $g$ (but it has no effect here, because $g$ is symmetric anyway);

 2) Move $g$ horizontally by $t$: this is a little abstract here; but if we
    align the $f$ and $g$ the way they were initially aligned, then we should
    get:

$$
f = [\dots 0, 0, 1, 1, \textbf{1}, \textbf{1}, 0, 0, 0, 0, \dots] \\
g = [\dots 0, 0, 0, 0, \textbf{2}, \textbf{2}, 2, 2, 0, 0, \dots] \\
$$

 3) Multiply all elements position by position and sum them all.

$$
  (f \ast g)(t) = 1 * 2 + 1 * 2 = 4
$$

You might have noticed how these operations may resemble dot-products.
You could have implemented them as:

$$
  (f \ast g)(t) = [1, 1] \dot [2, 2]
$$

This way, if you wanted to calculate the convolution for many
different values of $t$, you could just keep shifting the vector $g$.

_f

$$
t = 0 \\
f = [\dots 0, 0, 1, 1, 1, 1, 0, \dots] \\
g = [\dots 0, 0, 2, 2, 2, 2, 0, \dots] \\
(f \ast g)(t) = [1, 1, 1, 1] \dot [2, 2, 2, 2] = 8
\quad
t = 1 \\
f = [\dots 0, 0, 1, 1, 1, 1, 0, 0, \dots] \\
g = [\dots 0, 0, 0, 2, 2, 2, 2, 0, \dots] \\
(f \ast g)(t) = [1, 1, 1] \dot [2, 2, 2] = 6
\quad
f &= [\dots 0, 0, 1, 1, 1, 1, 0, 0, \dots] \\
g &= [\dots 0, 0, 2, 2, 2, 2, 0, 0, \dots] \\
(f \ast g)(t) &= [\dots 4, 6, 8, 6, 4, 2, 0, 0, \dots]
$$


Unfortunately, these are still vectors with an infinite number of
dimensions, which are hard to store in our limited storage computers.
But, well, we don't care about all those zeros there, so we could
just drop them:

$$
f = [0, 1, 1, 1, 1, 0, 0] \\
g = [0, 2, 2, 2, 2, 0, 0] \\
$$
_(As you can see, I kept some of the zeros. I could have removed them. It was my choice)_



The first of them is that
it is worth noting that very often
the functions $f$ and $g$ for which we want to calculate a
convolution are 0 in most of their domain.
Therefore it may be the case now that the non-zero parts of $f$ and
the shifted version of $g'$ do not even intersect, or only intersect
in some places (by "intersection", I mean that $g'$ and $f$ are non-zero
for the same $x$ inputs). Let's say they intersect in an
interval $[a, b]$.


This is precisely the kind of operation we need for image processing...


2D discrete convolutions
------------------------




