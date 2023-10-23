# PicoML
A machine learning framework for the Raspberry Pi Pico

--- 

## Introduction

More of a proof of concept rather than anything practical. Training a 7x7 neural net took literal minutes compared to my main computer which was a fraction of a second.
Nabbed the fraud payment data from the O'Reilly Machine Learning github, sampled it down to a few hundred rows so it can fit into the Pico's ram and trained it from there.
There is a test function to ensure the network is training properly. It trains the network to create an XOR function and runs it against an assert battery. It works 90% of the time but due to the threshold that is set sometimes it fails due to it being 98% sure instead of 99%.

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
