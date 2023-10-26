# PicoML
A machine learning framework for the Raspberry Pi Pico

--- 

## Introduction

More of a proof of concept rather than anything practical. Training a 7x7 neural net took literal minutes compared to my main computer which was a fraction of a second.
Nabbed the fraud payment data from the O'Reilly Machine Learning github, sampled it down to a few hundred rows so it can fit into the Pico's ram and trained it from there.
There is a test function to train the network to create an XOR function and run it against an assert battery. It works most of the time but due to the threshold that is set sometimes it fails due to it being 98% sure instead of at least 99%.

---
## Why?

MicroPython has a more limited set of packages compared to vanilla Python. Partially due to its age (9 years vs 32 years) partially because why on earth would you try to run pandas on a resource constrained system?

Eventually I'll start converting functions over to uLab (MicroPython's version of NumPy) but for right now I'm focusing on getting this usable.

---

## Future Plans

Eventually I'll get some more functions into the package.

* Backpropagation
* Linear/Logistic regression
* Support vector machine
* K-means clustering?

At a minimum get alot of the homebrew functions switched over to the uLab version so we can get a speed boost and maybe even a readability boost.

---

## Tested with 
* Raspberry Pi Pico W
* Thonny 4.1.2
* Windows 11 Home

---

## Sources
Code:
https://github.com/joelgrus/data-science-from-scratch/

Data:
https://github.com/oreilly-mlsec/book-resources
