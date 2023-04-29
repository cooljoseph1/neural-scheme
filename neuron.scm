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

         (output #f) ; A saved output (for memoization)
         
         ;; Function to do a forward pass through this neuron
         (forward-pass (lambda ()
                               (if output
                                   output
                                   (begin
                                     (set! inputs (map neuron:fire input-neurons))
                                     (set! output (apply forward-func inputs))
                                     output))))
         (backward-pass (lambda ()
                                (if gradients
                                    gradients
                                    (let* ((right-grad (apply + (map apply back-props)))
                                           (grad-list (backward-func inputs output right-grad)))
                                       (begin
                                         (set! gradients (list->vector grad-list))
                                         gradients)))))

         (reset! (lambda () (if output (begin                ; We don't need to reset if we already are reset
                                         (set! inputs #f)
                                         (set! gradients #f)
                                         (set! output #f))
                                      #f)))

         (set-inputs! (lambda inps (begin
                                     (set! inputs inps)
                                     (set! output (apply forward-func inputs)))))
                                     
         ;; Set the loss on the right side of this neuron (i.e. negative of the derivative w.r.t. the output) and use that to infer this neuron's gradient
         (set-loss! (lambda (loss) (begin
                                     (if inputs
                                         #t
                                         (error "Input not set for this neuron. Did you forget to run a forward pass before back propagating?"))
                                     (if output
                                         #t
                                         (error "Output not set for this neuron. Did you forget to run a forward pass before back propagating?"))
                                     (let* ((grad-list (backward-func inputs output (- loss))))
                                       (begin
                                         (set! gradients (list->vector grad-list))
                                         gradients))))))
                                      

         ;; Setter function for the list of input neurons
         (define (set-input-neurons! new-input-neurons) (set! input-neurons new-input-neurons))

         ;; Add a back propagation function to our list of back propagation functions
         (define (add-back-prop! back-prop)
                 (set! back-props (cons back-prop back-props)))

    (list forward-pass backward-pass set-input-neurons! add-back-prop! reset! set-inputs! set-loss!)))

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

(define (neuron:reset-getter neuron)
  (caddr (cddr neuron)))

;;; Reset to prepare for a new forward pass
(define (neuron:reset! neuron)
  ((neuron:reset-getter neuron)))

(define (neuron:raw-input-setter neuron)
  (caddr (cdddr neuron)))

;;; Directly set the inputs to a neuron
(define (neuron:set-raw-inputs! neuron inputs)
  ((neuron:raw-input-setter neuron) inputs))


(define (neuron:raw-loss-setter neuron)
  (cadddr (cdddr neuron)))

;;; Directly set the loss on the right side of the neuron
(define (neuron:set-raw-loss! neuron loss)
  ((neuron:raw-loss-setter neuron) loss))

;;; Join a neuron together with its input neurons by binding the input neurons to the neuron's inputs and
;;; linking together their back propagation
(define (neuron:join! input-neurons neuron)
  (neuron:set-input-neurons! neuron input-neurons)
  (map neuron:add-back-prop! input-neurons
                            (map (lambda (index)
                                         (lambda ()
                                                  (vector-ref (neuron:grad neuron) index)))
                                 (iota (length input-neurons)))))


#| Primitive types of neurons |#

#| Arithmetic neurons |#
;;; Add
(define (add-forward . args)
  (apply + args))

(define (add-backward inputs output grad)
  (map (lambda (x) grad) inputs))

;;; Make an add neuron
(define (make-add-neuron)
  (make-neuron add-forward add-backward))

;;; Multiply
(define (mult-forward x y)
  (* x y))

(define (mult-backward inputs output grad)
  (list (* grad (cadr inputs)) (* grad (car inputs))))

;;; Make a multiply neuron (exactly two inputs)
(define (make-mult-neuron)
  (make-neuron mult-forward mult-backward))

(define (exp-forward x)
  (exp x))
(define (exp-backward inputs output grad)
  (list (* grad (exp-forward (car inputs)))))

(define (make-exp-neuron)
  (make-neuron exp-forward exp-backward))

(define (div-forward x y)
  (/ x y))

(define (div-backward inputs output grad)
  (map (lambda (x) (* x grad)) (list (/ 1 (cadr inputs)) (- (/ (car inputs) (* (cadr inputs) (cadr inputs)))))))

(define (make-div-neuron)
  (make-neuron div-forward div-backward))

(define (log-forward x)
  (log x))

(define (log-backward inputs output grad)
  (list (* grad (/ 1 (car inputs)))))

(define (make-log-neuron)
  (make-neuron log-forward log-backward))

#| Activation neurons |#
;;; ReLU
(define (relu-forward x)
  (max 0 x))

(define (relu-backward inputs output grad)
  (if (> output 0)
      (list grad)
      (list 0)))

(define (make-pow-neuron y)
  (make-neuron (lambda (x) (expt x y)) (lambda (inputs output grad) 
  (list (* grad y (expt (car inputs) (- y 1)))))))
;;; Make a ReLU neuron (single input one output)
(define (make-relu-neuron)
  (make-neuron relu-forward relu-backward))

;;; Sigmoid
(define (sigmoid-forward x)
  (/ 1 (+ 1 (exp (- x)))))

(define (sigmoid-backward inputs output grad)
  (list (* grad output (- 1 output))))

;;; Make a sigmoid neuron (single input one output)
(define (make-sigmoid-neuron)
  (make-neuron sigmoid-forward sigmoid-backward))

;;; tanh
(define (tanh-forward x)
  (/ (- 1 (exp (- (* 2 x)))) (+ 1 (exp (- (* 2 x))))))

(define (tanh-backward inputs output grad)
  (list (- grad (* grad output output))))

;;; Make a sigmoid neuron (single input one output)
(define (make-tanh-neuron)
  (make-neuron tanh-forward tanh-backward))

#| Misc. other neurons |#
(define (identity-forward x) x)

(define (identity-backward inputs output grad) (list grad))

;;; Function that returns an identity neuron
(define (make-identity-neuron)
  (make-neuron identity-forward identity-backward))

(define activation-list (list (list 'sigmoid make-sigmoid-neuron)
                              (list 'relu make-relu-neuron)
                              (list 'tanh make-tanh-neuron)))

;;; Neuron that always fires the same value
(define (make-input-neuron value)
  (make-neuron (lambda x value) (lambda (inputs output grad) 0)))

;;; return (setter function, neuron)
(define (make-neuron-controllable)
  (let ((val 0))
    (list (lambda (x) (set! val x)) (make-neuron (lambda x val) (lambda (inputs output grad) 0)))))


