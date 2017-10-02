---
use_math: true
layout: post
title: Arrays and Their Multiple Facets
---

In
[my first blog post on Convolutions](https://vaulttech.github.io/2017/08/12/what-are-convolutions/)
(no need to go read there: this blog post is supposed to be
"self-contained")
I discusssed a little about how it would be a good idea to reinterpret
the discretized version of the 1D function $f$ as a vector with an
infinite number of dimensions. Basically, the only difference between
the two ways of viewing this "list of numbers" was that the vector
lacked a "reference point", _i.e._, the $t$ we had there. Because $f$
was a
very nice type of function that was non-zero only for a certain range
of $t$'s, we found a way to get this reference point back by dropping
the rest of $f$ where $f$ was always zero.

In this blog post, I want to talk about yet another way in which we
can look at a vector (and, consequently, at a function $f$). In the
next few sections, I will recapitulate the ideas presented in
[the blog post on Convolutions](https://vaulttech.github.io/2017/08/12/what-are-convolutions/),
explain the other interpretation of vectors, and show how it may be
useful when training a classifier.


Arrays Can Be Reinterpreted As Discrete Functions
-------------------------------------------------

Let's recapitulate what we learned in the previous blog post. In the
example, I had a signal $f$ that looked like the following:

![The original f function](/public/...)

Because we wanted to avoid calculating an integral (the calculation
of the convolution, which was the problem we wanted to solve,
required the solution of an integral), and because we
were not dramatically concerned with numeric precision, we concluded
it would be a good approximation to just use a discrete version of
this signal. We therefore sampled only certain evenly spaced points
from this function, and we called this process "discretization":

![The discretized f](/public/...)

_(In our original setting, $f$ was a function that turned out to be
composed by non-zero values only in a small part of its domain. The
rest was only zeros, extending vastly to the right and to the left
of that region. This was convenient for our convolutions, and will
be convenient too for our discussion below, although most of the
ideas presented below are going to still work if we drop this
assumption.)_

I would like to introduce some names here, so that I can refer to
things in a more unambiguous way. Let $f_{discretized}$ be the newly
created function, that came into existence after we sampled several
points from $f$, all of which are evenly spaced. Additionally, let
us call $s$ the space between each sample. For the purposes of
this blog post, we will consider we have any arbitrary $s$. It does
not really matter how big or small $s$ is, as long as you (as a
human being) feel that the new discrete function you are defining
resembles well enough (based on your own notion of "enough") the
original $f$. If you choose an $s$ that is too large, you might
end up missing all non-zero points of $f$ (or taking only
one non-zero point, depending on where you start). If your
$s \to 0$, then you have back the continuous function, and your
discretization had basically no effect.

Your new function $f_{discretized}$ now could be seen as a vector
composed of mostly zeros, except for a small region:

$$
f_{discretized} = [\dots 0, 0, 1, 1, 1, 1, 0, 0, \dots]
$$

Because this is an infinite array, it is hard to know exactly where
it "starts" (or where it "ends"). In the introduction to this post I
said this was a "problem", and we had solved it by dropping
the two regions composed exclusively by zeroes:

$$
f_{discretized} = [1, 1, 1, 1]
$$

Of course, we could have retained some of the zeros, if it was for
any reason convenient to us. It doesn't matter much. The main idea
here is that we now have a convenient way to represent functions
compactly through vectors. This also means that anything that works
for vectors (dot products, angles, norms) also should have some
interpretation for discrete functions. Think about it!


#### Disclaiming Interlude

To say the truth, I don't think that the lack of a "reference point",
as I said before, is a problem at all. From a
"maths" perspective, we could solve this by adopting literally any
element as our "start", and from there we can index all other
elements. We could even conveniently choose the element that
corresponds to our $t = 0$, and it is almost as if we had $f$ back.
Mathematicians are quite used to deal with "infinity", and
it these seem quite reasonable ideas.

Other human beings, however, would probably not have the same ease,
and our machines have unfortunately a limited amount of memory. We
would like to keep in our memory only the things we actually care
about... and we don't care a lot about zeros: they kill any number
they multiply with, and work as an identity after the sum.


Arrays Can Be Reinterpreted As Distributions
--------------------------------------------

It is very likely that, just by reading the heading of this section,
you already got everything you need to know. There is no magic
insight in here: I just intend to go through the ideas slowly and
make it clear why (and, in some ways, how) the heading is true.
If you already got it, I would invite you to skip to the next section,
that tries to show examples when the multiple facets of vectors are
useful. If you stick to me, however, I hope this section may be
beneficial.


### What is a Distribution?

When I had a course on Statistics in my Bachelor, it was really bad.
At the time of the exam, it seemed I should be much more concerned
with how to round the decimal numbers after the
[comma](https://en.wikipedia.org/wiki/Decimal_mark), than with the
actual concepts I was supposed to have learnt.
As a consequence, I didn't understand much of statistics when I
started with Machine Learning and it took me a great deal of
self-studying to realize some of the things in this blog post.

One of these things was the meaning of the word _distribution_. This
is for me a tricky word, and to be fair I might still miss some of its
theoretical details (I just went to Wikipedia, and
[the article on the topic](https://en.wikipedia.org/wiki/Probability_distribution)
seems so much more complicated than I'd like it to be). For our
purposes here, I will consider a _distribution_ any function that
satisfies the following two criteria:

 1. It is composed exclusively by positive numbers
 2. The area below the curve sums up to 1

_(For the avid reader: I am avoiding the word "integral"
because I don't want to bump into "the integral of a point", that is
tricky and unnecessary here)_

There is one more important element to be discussed about
distributions: any distribution $d$ is a function of one of more
_random variables_. These variables represent the thing we are trying
to find the probability for. For example, they might be the _height_
of the people in a population, the _time_ people take to read a
sentence, or the _age_ of people when they lose their first tooth.


### On Continuous Distributions

If a random variable can assume any value in a continuous interval,
then a distribution over this random variable is also defined
over all values of this interval. For example, if my random variable
represents the height of a person (and let's assume this height can be
anything between 10cm to 3m), then a distribution over this variable
should be defined for every single element in the interval
[10cm, 300cm].

(While writing this text, I found
[this applet](http://www.shodor.org/interactivate/activities/NormalDistribution/)
that show a continuous Gaussian distribution and a histogram that
resembles it. It will be useful for the discussion to follow)

However, (and although it may not have been obvious by the discussion
so far) the one
trick that allows us to represent function as vectors is the fact
that we discretize them first: they then become "countably infinite",
which means that each element in the function can be indexed by an
integer number. Since Continuous Distributions _are_ continuous
functions [of a random variable], this is also valid for them.


These discretized versions of our continuous distributions are
basically histograms (and the size of the bins of the histogram
work exactly like the space $s$ between each sample point).

Now imagine what would happen if 

If our distributions are non-negligible only for a certain range of our
random variable (like the Gaussian function, where ~95% of the area
below the curve is between the mean and two standard deviations
around it), then we have a scenario that looks pretty much like the
one we had when discussing Convolutions, and it should be clear how
to transform this function into a finite vector.


### On Discrete Distributions

Discrete Distributions are already 


3. Arrays can be reinterpreted as distributions

 * When we talk about distributions, we often think of densities

 * Discrete distributions

 * Representing them as vectors

 * Getting a distribution out of a non-sum to 1

 * This naturally brings up one-hot encoding






4. How is this useful?

  * Entropy

  * Categorical cross-entropy

  * For example, we could use the cosine between two functions $a$ and $b$ as a measure of their
difference. 

  * KL-Divergence


