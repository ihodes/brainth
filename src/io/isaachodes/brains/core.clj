(ns io.isaachodes.brains.core
  "Provides a functional backpropagation implementation for deep-learning neural
   networks.")


(def ^:dynamic *learning-rate* 0.2)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;; Small helpers
(def p partial)
(def zip (partial map list))
(defn random [] (Math/random))
(defn update-first [f [a b]] [(f a) b])
(defn update-second [f [a b]] [a (f b)])


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;; Linear Algebra helpers
(def sum (partial reduce +))

(defn dot-product
  [xs ys]
  (sum (map * xs ys)))

(defn matrix
  "Returns a seq of columns with initial value set by calling init."
  [init [row col]]
  (take col (repeatedly #(repeatedly row init))))

(defn T
  "Transposes matrix M."
  [M]
  (map (fn [idx] (map nth M (repeat idx)))
       (range (count (first M)))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;; Backprop Backup
(defn sigma
  "Sigma function for step function approximation."
  [x]
  (Math/tanh x))

(defn sigma'
  "Derivative of the sigma function."
  [x]
  (- 1 (Math/pow (sigma x) 2)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;; Backpropagation & training proper

(defn- -run
  "Executes the network, returning a sequence of tuples of the form
   [node-vals node-activations]. The result of the run is the last
   `node-activations` value; the result that the network expects."
  [input network-weights]
  (let [execute-layer (fn [[ins acts] weights]
                        (let [in'   (map (partial dot-product acts) weights)
                              acts' (cons 1 (map sigma in'))] ;; add bias node
                          [in' acts']))
        inputs+activations (reductions execute-layer
                                       ['() (seq input)]
                                       network-weights)
        [output-ins output-as] (last inputs+activations)]
    (concat (butlast inputs+activations) [[output-ins (rest output-as)]])))

(defn backpropagate
  "Returns a new collection of network weights after backpropagating errors
   from the classification (as compared to the expected classification) 
   of the given inputs with a momentum, alpha.

   This is my functional adaptation of the classic Norvig & Russel algorithm for
   backpropagation."
  [input expected network-weights]
  (let [inputs+activations      (-run input network-weights)
        [output-in output-acts] (last inputs+activations)
        r-inputs+activations    (reverse (butlast inputs+activations))
        r-network-weights       (reverse network-weights)
        
        ;; Here is the std error between expected and the actual output layer.
        output-layer-deltas (map (fn [i a e] (* (sigma' i) (- e a)))
                                 output-in output-acts expected)

        ;; Here we backpropagate the error from the output layer through
        ;; all of the hidden layers, starting from the last hidden layer.
        r-layers-deltas (reductions (fn [deltas [weights ins]]
                                      (let [weighted-errors
                                            (map (partial dot-product deltas)
                                                 (T weights))]
                                        (map (fn [i s] (* (sigma' i) s))
                                             ins weighted-errors)))
                                    output-layer-deltas
                                    (zip r-network-weights (map first r-inputs+activations)))

        ;; Find new network weights for each layer (back-to-front).
        new-network-weights (map (fn [layer-weights [_ as] deltas]
                                   (map (fn [node delta]
                                          (map (fn [weight a]
                                                 (+ weight (* *learning-rate* a delta)))
                                               node as))
                                        layer-weights deltas))
                                 r-network-weights r-inputs+activations r-layers-deltas)]
    (reverse new-network-weights)))

(defn backpropagate-set
  "Runs backpropagate on the training set, returning the adjusted network weights."
  [inputs expecteds network-weights]
  (reduce (fn [ws classified]
            (backpropagate (first classified) (second classified) ws))
          network-weights
          (zip inputs expecteds)))

(defn train
  "Returns an infinite sequence of network weights, each one trained one more time than
   the last on the entire training set."
  [inputs expecteds network-weights]
  (iterate (partial backpropagate-set inputs expecteds) network-weights))

(defn run
  "Runs the network on the given input, returning the sigmoid response from the 
   output nodes."
  [input network-weights]
  (last (last (-run input network-weights))))
 
(defn answer 
  "Applies a simple threshhold step function to the results of running the network on
   the given input, returning definite classifications (0 or 1)."
  [input network-weights]
  (map #(if (> 0.5 %) 0 1)
       (run input network-weights)))

(defn error
  "The standardized error for the given input vector."
  [input expected network-weights]
  (sum (map (fn [in exp] (* 0.5 (Math/pow (- exp in) 2)))
            (run input network-weights) expected)))

(defn set-error
  "The test set error."
  [inputs expecteds network-weights]
  (sum (map #(error %1 %2 network-weights) inputs expecteds)))

(defn initialize-network-weights
  "Randomly initializes the weights of a neural network with layers of size
   specified by sizes, respectively.

   Each row of each matrix (nodes in layer + bias (row) by nodes in next layer (col))
   contains the weights of the edges from that node to all of the nodes in the
   next layer."
  [& sizes]
  (map (partial matrix random)
       (map (partial update-first inc) (partition 2 1 sizes))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;; Testing & Appendix

(comment (def +inputs+ [[1 1] [1 0] [0 1] [0 0]]) ;; logic gate inputs
         ;; AND, OR, XOR, NOR gates
         (def +expecteds+ [[1, 1, 0, 0] [0, 1, 1, 0] [0, 1, 1, 0] [0, 0, 0, 1]])

         (def +untrained-logic-network+
           (initialize-network-weights 2 3 4))

         (def +trained-logic-network+
           (nth (train +inputs+ +expecteds+ +untrained-logic-network+) 100))

         (set-error +inputs+ +expecteds+ +untrained-logic-network+)
         (set-error +inputs+ +expecteds+ +trained-logic-network+))


#_(defn simple-backpropagate
  "Simple, single-hidden-layer algorithm. Useful for readability."
  [input expected network-weights]
  (let [rate       *learning-rate*
        input-as   (add-bias (seq input))
        hidden-in  (map (p dot-product input-as) (first network-weights))
        hidden-as  (add-bias (map sigma hidden-in))
        output-in  (map (p dot-product hidden-as) (second network-weights))
        output-as  (map sigma output-in)

        input-to-hidden-weights (first network-weights)
        hidden-to-output-weights (second network-weights)

        output-deltas (map (fn [i a e] (* (sigma' i) (- e a)))
                           output-in output-as expected)

        inter-output-sums (map (partial dot-product output-deltas)
                               (T hidden-to-output-weights))
        hidden-deltas (map (fn [i s] (* (sigma' i) s))
                           hidden-in inter-output-sums)

        new-h-o-weights (map (fn [output-node output-delta]
                               (map (fn [weight hidden-a]
                                      (+ weight (* rate hidden-a output-delta)))
                                    output-node hidden-as))
                             hidden-to-output-weights output-deltas)


        new-i-h-weights (map (fn [hidden-node hidden-delta]
                               (map (fn [weight input-a]
                                      (+ weight (* rate input-a hidden-delta)))
                                    hidden-node input-as))
                             input-to-hidden-weights hidden-deltas)]
    (list new-i-h-weights new-h-o-weights)))
