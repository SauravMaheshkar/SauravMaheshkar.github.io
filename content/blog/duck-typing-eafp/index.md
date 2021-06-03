---
date: "2020-12-12"
draft: false
excerpt: Brief Introduction to Duck Typing and EAFP (Programming Methodology)
categories:
  - Programming
  - Python
subtitle: ""
title: Introduction to Duck Typing and EAFP
weight: 1
---

Duck-Typing is a extremely useful programming style, which truly makes python awesome. It enables us to "ignore" the object type and rather just check if the object contains the function or not.

Famously referred in the [python documentation](https://docs.python.org/3/glossary.html#term-duck-typing) as:

> "If it looks like a duck and quacks like a duck, it must be a duck"

If the codebase is well defined, this allows for flexibility by allowing polymorphic substitution. It allows us to avoid stuff like `type()` and `isinstance()`.

Let's consider this example. We create two classes, Duck and Octocat. Because Octocat is awesome, let's assume it's capable of quacking.
```python
class Duck:
  def quack(self):
    print("Quack, Quack")

class Octocat:
  def meow(self):
    print("Github is awesome")
  def quack(self):
    print("Quack, Quack")
```

Now let's create a function `make_it_quack()`, Let's assume we don't know about Duck Typing and we need to write a function that checks whether the object which we pass into the function is of type `Duck` and if it is we invoke the `quack()` function

```python
def make_it_quack(object):
  if isinstance(object, Duck):
    object.quack()
  else:
    print("Only duck's can quack")
```

If we pass a Duck object (say `d`) our output will be:

```python
make_it_quack(d)
>>> Quack, Quack
```

If we pass a Octocat object (say `o`) our output will be:

```python
make_it_quick(o)
>>> Only duck's can quack
```

In a package, there can be tons and tons of objects, we can't type check for every possible object type (But that is what people would do when python didn't exist!!), we can just do the following:

```python
def make_it_quack(object):
  object.quack()

make_it_quack(o)

>>> Quack, Quack
```

This is the essence of *Duck-Typing*.

---

**Look before you Leap(LBYL)**: This is coding style which explicitly tests for pre-conditions before making calls or lookups. It's characterised by the presence of many `if` statements.

**Easier to ask for forgiveness than permission**: This is a common python coding style, which just assumes the existence of valid keys/attributes and only catches exceptions if this assumptions prove false. This makes the code much cleaner and is usually characterised by the presence of `try` and `except` statements.

Let's assume we're analysing some `metadata` dictionary and we need to extract a particular key value

## The LBYL way

```python
metadata = {"day": "Monday",
            "month": "December",
            "visited":True
           }
if "day" in metadata and "month" in metadata and "visited" in metadata:
  visit = metadata["day"]
else:
  print("Missing key")
```

## The EAFP way

```python
try:
    visit = metadata["visited"]
except KeyError as e:
    print("{} not found".format(e))
```

It's pretty intuitive why EAFP is better than the LBYL approach.