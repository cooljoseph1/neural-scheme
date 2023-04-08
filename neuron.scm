;;;; Implement neurons--the basic building blocks of a network.

#|
A neuron always has the following properties:
1. Exactly one output
2. Any number of inputs
|#

;;; Make a basic neuron. It has a forward function that takes in arbitrarily many inputs
;;; and returns a single output, and a backward function that takes in a single output gradient
;;; and returns arbitrarily many input gradients.


;;; Create a neuron. This takes two arguments and returns a neuron.
;;; Args:
;;;  forward: A function that takes any number of arguments as inputs and outputs a single scalar
;;;  backward: A function that takes in a list of inputs to the neuron and a gradient at the output
;;;            and returns a list of output gradients

(define (make-neuron forward backward)
  (let* ((bound-backward (lambda grad (error "Backward graph not created!")))
         (bind! (lambda (inputs)
                        (set! bound-backward
                              (lambda (grad) (backward inputs grad)))))
         (forward-pass (lambda inputs
                               (begin
                                 (bind! inputs)
                                 (apply forward inputs))))
         (backward-pass (lambda (grad) (bound-backward grad))))
    (list forward-pass backward-pass)))
 
  (list forward backward))

(define (get-forward neuron)
  (car neuron))

(define (get-backward neuron)
  (cadr neuron))

(define (stored-values neuron)
  (caddr neuron))
  


(define (forward neuron inputs)
  (set-caddr!)
  (apply (get-forward neuron) inputs))

(define (backward neuron output)
  ((get-backward neuron) (stored-values neuron) output))



#| Discard this?


(define (weight-object initial-weight)
  (define weight initial-weight)
  (cons (lambda () weight)                              ; weight getter
        (lambda (new-weight) (set! weight new-weight)))) ; weight setter
  

(define (weight-getter weight-object)
  (car weight-object))
  
(define (weight-setter weight-object)
  (cdr weight-object))
  


(define (make-weight-neuron initial-weight)
  (define weight (weight-object ))
  (make-neuron (lambda () weight)
               (lambda (output) '())))
|#



(define (add-forward . inputs)
  (apply + inputs))

(define (add-backward inputs grad)
  (map (lambda (x) grad) inputs))

(define (relu-forward input)
  (if (< input 0)
    0
    input))

(define (relu-backward inputs grad)
  (if (< (car input) 0)
    0
    grad))

(define (mult-forward i1 i2)
  (* i1 i2))

(define (mult-backward inputs grad)
    (list (* grad (cadr inputs)) (* grad (car inputs))))

(define (make-mult-neuron)
  (list mult-forward mult-backward '()))


(define (make-backward function inputs)
  (lambda (grad)
    (function inputs grad)))

(define (forward-pass neuron inputs))

(define (backward-pass neuron grad))




