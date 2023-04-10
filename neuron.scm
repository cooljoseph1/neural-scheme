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


#| Make a neuron that adds together all its inputs |#
(define (add-forward . args)
  (+ args))

(define (add-backward inputs grad)
  (map (lambda (x) grad) inputs))

;;; Function that returns a neuron that adds together its inputs
(define (add-neuron)
  (make-neuron add-forward add-backward))

#| Make a neuron that multiplies together exactly two inputs
(define (mult-forward x y)
  (* x y))

(define (mult-backward inputs grad)
  (list (* grad (cadr inputs)) (* grad (car inputs))))

(define (mult-neuron)
  (make-neuron mult-forward mult-backward))

