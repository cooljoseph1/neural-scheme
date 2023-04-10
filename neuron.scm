;;;; Implement neurons--the basic building blocks of a network.

#|
A neuron always has the following properties:
1. Exactly one output
2. Any number of inputs
|#

;;; Make a neuron given the forward function and backward function.
;;; The forward function takes any number of inputs and returns a single value.
;;; The backward function takes a single value and returns a list of gradients with
;;; respect to the inputs.
(define (make-neuron forward-func backward-func)
  (let* ((input-neurons '())  ; A list of all neurons feeding into this neuron. Used for forward pass
         (back-props '()) ; A list of back propagation functions that need to accumulate at the right side of this neuron
         
         (inputs #f) ; A list of inputs that the input neurons fired with (used in back propagation)
         (gradients #f) ; A vector of gradients of the loss with respect to each of the input neurons (in order)
         
         ;; Function to do a forward pass through this neuron
         (forward-pass (lambda ()
                               (if inputs
                                   (apply forward inputs)
                                   (begin
                                     (set! inputs (map neuron:fire input-neurons))
                                     (apply forward inputs)))))
         (backward-pass (lambda ()
                                (if gradients
                                    gradients
                                    (let* ((right-grad (apply + (map apply back-props)))
                                           (grad-list (backward-func inputs right-grad)))
                                       (begin
                                         (set! gradients (list->vector grad-list))
                                         gradients)))))

         (reset! (lambda () (begin
                              (set! inputs #f)
                              (set! gradients #f))))

         ;; Setter function for the list of input neurons
         (define (set-input-neurons! new-input-neurons) (set! input-neurons new-input-neurons))

         ;; Add a back propagation function to our list of back propagation functions
         (define (add-back-prop! back-prop) (set! back-props (cons back-prop back-props)))

    (list forward-pass backward-pass set-input-neurons! add-back-prop!)))

(define (neuron:get-forward neuron)
  (car neuron))

(define (neuron:fire neuron)
  ((neuron:get-forward neuron)))

(define (neuron:get-backward neuron)
  (cadr neuron))

(define (neuron:grad neuron)
  ((neuron:get-backward neuron)))

(define (neuron:inputs-setter neuron)
  (caddr neuron))

(define (neuron:set-input-neurons! neuron input-neurons)
  ((neuron:inputs-setter neuron) input-neurons))

(define (neuron:back-prop-adder neuron)
  (cadddr neuron))

(define (neuron:add-back-prop! neuron back-prop)
  ((neuron:back-prop-adder neuron) back-prop))

;;; Join a neuron together with its input neurons by binding the input neurons to the neuron's inputs and
;;; linking together their back propagation
(define (neuron:join! input-neurons neuron)
  (neuron:set-input-neurons! neuron input-neurons)
  (map neuron:add-back-prop! input-neurons
                            (map (lambda (index)
                                         (lammbda ()
                                                  (vector-ref (neuron:grad neuron) index)))
                                 (iota (length input-neurons)))))


;;; Make a basic neuron. It has a forward function that takes in arbitrarily many inputs
;;; and returns a single output, and a backward function that takes in a single output gradient
;;; and returns arbitrarily many input gradients.

;;; Create an activation function
;;; Args:
;;;  f: The activation function.
;;;  df: A function that takes in

(define (relu)
  (let* ((bound-df (lambda ))
         (f (lambda (x) (max 0 x))

;;; Create a neuron. This takes two arguments and returns a neuron.
;;; Args:
;;;  activation: A (nonlinear) function that takes a single input and returns a single output
;;;  activation-derivative: The partial derivative of the output with respect to the input

(define (make-neuron forward-func backward-func)
  (let* ((input-neurons '())  ; A list of all neurons feeding into this neuron. Used for forward pass
         (output-neurons '()) ; A list of all neurons this neuron's fire value is sent to. Used for backpropagation
         ;; Where the backward function with bound inputs is stored
         (bound-backward (lambda grad (error "Backward graph not created!"))) ; By default it throws an error because it shouldn't be called before a forward pass

         ;; Helper function to bind inputs to the backward function
         (bind! (lambda (inputs)
                        (set! bound-backward
                              (lambda (grad) (backward inputs grad)))))
         
         ;; Function to do a forward pass through this neuron
         (forward-pass (lambda ()
                          (let ((inputs (map neuron:forward input-neurons)))  ; Get the fire values of the input neurons
                            (begin
                              (bind! inputs)                                  ; Bind inputs for back propagation
                              (apply forward inputs))))
         ;; Function to do a backward pass through this neuron (returning the gradients for its inputs)
         (backward-pass (lambda () (bound-backward grad))))
    (cons forward-pass backward-pass)))

(define (neuron:get-forward neuron)
  (car neuron))

(define (neuron:get-backward neuron)
  (cdr neuron))
  
(define (neuron:forward neuron inputs)
  (apply (neuron:get-forward neuron) inputs))

(define (neuron:backward neuron output)
  ((neuron:get-backward neuron) output))

#| Now make some primitive neurons |#
;;; Neuron for adding things together
(define (add-forward . inputs)
  (apply + inputs))

(define (add-backward inputs grad)
  (map (lambda (x) grad) inputs))

(define +-neuron
  (lambda () (make-neuron add-forward add-backward)))

;;; Neuron for multiplying two things together
(define (mult-forward x y)
  (* x y))

(define (mult-backward inputs grad)
  (let ((x (car inputs))
        (y (cadr inputs)))
    ((* grad y) (* grad x))))

(define *-neuron
  (lambda () (make-neuron mult-forward mult-backward)))

;;; Relu activation function
(define (relu-forward input)
  (max 0 input))

(define (relu-backward inputs grad)
  (if (< (car inputs) 0)
    0
    grad))

(define relu-neuron
  (lambda () (make-neuron relu-forward relu-backward)))


