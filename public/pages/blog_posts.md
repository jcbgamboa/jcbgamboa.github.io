---
layout: page
title: All Blog Posts
---

I noticed it was becoming hard to find all my blog posts after a while, so I
thought of creating this list to access them easily and keep track of how often
I post. Hopefully this is useful for the interested visitor too :-)

<div>
    {% for post in site.posts %}    
        <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
        <p><small><strong>{{ post.date | date: "%B %e, %Y" }}</strong></small></p>
        <!-- This post listing code snippet was taken from:
        https://gist.github.com/erjjones/1998382 -->
    {% endfor %}
<div>

