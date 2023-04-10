;;;; Modules
;;; Modules are how you combine together to make neural *nets*. They are composed of
;;; neurons and submodules. Modules take care of things like actually running inputs
;;; through a network, memoization, how to attach parameters, etc. They do not on
;;; Modules are the basic building blocks of a neural network. They take in inputs
;;; and give outputs through a forward pass. They can also take in gradients at their
;;; output neurons and give out gradients at their input neurons in their backward pass,
;;; saving internal gradients along the way.

;;; First, load some packages
(load "utils.scm")
(load "neuron.scm")

#| A module needs to implement two things:
(1) A forward pass
(2) A backward pass
|#

;;; Make a module from a forward function and a backward function.
;;; The forward function should take any number of inputs and return a list of outputs.
;;; The backward function should take the above number of outputs and return a list of the above number of inputs.
;;; TODO: Memoize things
(define (make-module inputs output-neurons)
  (let ((forward (lambda args)
  (cons forward backward))

;;; Return the (memoized) forward procedure of a module
(define (module:get-forward module)
  (car module))

;;; Return the (memoized) backward procedure of a module
(define (module:get-backward module)
  (cdr module))

#| Implement various types of modules |#
;;; Turn a neuron into a module
(define (neuron->module neuron)
  (let ((forward (neuron:get-forward neuron))
        (backward (neuron:get-backward neuron)))
    (make-module (lambda args (list (apply forward args))) ; Wrap the output in a list for consistency between different types of modules
                 backward)))

;;; Create a module that is a pair of submodules in series
;;; The graph flow for the forward pass is inputs --> module1 --> module2 --> outputs
(define (module:combine-series module1 module2)
  (make-module (list-compose (module:get-forward module2)
                        (module:get-forward module1))
               (list-compose (module:get-backward module1)
                        (module:get-backward module2))))


;;; Create a module that is a list of submodules in parallel
(define (module:combine-parallel module1 module2)
  (let* ((arity-first -1)
         (forward1 (module:get-forward module1))
         ;; Wrap the first module's forward method to keep track of how many arguments
         ;; we need to send to its backwards pass (and how many need to be sent to the
         ;; other module)
         (wrapped-forward1 (lambda args
                                  (begin
                                    (set! arity-first (length args))
                                    (apply forward1 args))))
         (forward (lambda args (append (apply wrapped-forward1 args)
                                       (apply (module:get-forward module2) args))))
         ;; Split the backwards args so half go to the 
         (backward (lambda args (append (apply wrapped-forward1 args)
                                       (apply (module:get-forward module2) args))))

