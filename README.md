# Brains

#### A deep-learning functional, idiomatic, neural-network in Clojure.

This project provides an idiomatic and functional implementation of a deep-
learning neural network using backpropagation learning, as well as additional
functions to easily use these neural networks.

The focus is on legibility, not on execution speed. Using a mature library like
[Encog](http://www.heatonresearch.com/encog) is recommended; a nice-looking
Clojure wrapper can be found at [Enclog](https://github.com/jimpil/enclog).
While I've added a fair few comments throughout the code, it may only be enough
to guide someone who already has a good grasp on backpropagation.

My backpropagation algorithm is based on Norvig and Russell's imperative
algorithm described in their excellent textbook on artificial intelligence,
["Artificial Intelligence: A Modern Approach"](http://aima.cs.berkeley.edu/).

For usage, see the below example code for training a net to learn AND, OR, XOR
and NOR logic gates, and inspecting error before and after.

```clojure
(def +inputs+ [[1 1] [1 0] [0 1] [0 0]])

;; AND, OR, XOR, NOR gates
(def +expecteds+ [[1, 1, 0, 0] [0, 1, 1, 0] [0, 1, 1, 0] [0, 0, 0, 1]])

(def +untrained-logic-network+
(initialize-network-weights 2 3 4))

(def +trained-logic-network+
(nth (train +inputs+ +expecteds+ +untrained-logic-network+) 100))

(set-error +inputs+ +expecteds+ +untrained-logic-network+)
(set-error +inputs+ +expecteds+ +trained-logic-network+)
```

NB: Momentum was purposefully left out. Though adding it would be trivial, it obfuscates the code to keep a copy of the weight changes from the previous round. 
