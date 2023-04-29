;;;; Modules
;;; Modules are how you combine together to make neural *nets*. They are composed of
;;; neurons and submodules. Modules take care of things like actually running inputs
;;; through a network, memoization, how to attach parameters, etc. They do not on
;;; Modules are the basic building blocks of a neural network. They take in inputs
;;; and give outputs through a forward pass. They can also take in gradients at their
;;; output neurons and give out gradients at their input neurons in their backward pass,
;;; saving internal gradients along the way.

;;; First, load some packages
(load "neuron.scm")
(load "param.scm")

;;; A module has the following three things:
;;; 1. A number of input neurons that can be linked going into
;;; 2. A number of output neurons that can be linked going out of
;;; 3. A list of parameters that can be modified
;;; 4. A function to reset the network to prepare for a new forward pass & back pass

(define (make-module input-neurons output-neurons parameters reset-function)
  (list input-neurons output-neurons parameters reset-function))

;;; Return the input neurons of the module
(define (module:get-input-neurons module)
  (car module))

;;; Return the output neurons of the module
(define (module:get-output-neurons module)
  (cadr module))

;;; Return the parameters of the module
(define (module:get-params module)
  (caddr module))

;;; Return the reset function of the module
(define (module:get-reset-function module)
  (cadddr module))

;;; Reset the module to prepare for a forward pass
(define (module:reset! module)
  ((module:get-reset-function module))
  (map param:zero-grad! (module:get-params module)))

;;; Given a list of inputs, run it through the module. (Note: This is mostly only so complicated because we have to attach temporary inputs)
(define (module:forward module inputs)
  ;; Step 1: Check to make sure the number of inputs is correct
  (let ((input-neurons (module:get-input-neurons module)))
    (if (= (length input-neurons) (length inputs))
        #t
        (error "Did not supply correct number of inputs to the module" module inputs (length module) (length inputs)))
    ;; Step 2: Set the inputs for those neurons to the provided inputs
    (map neuron:set-raw-inputs! input-neurons inputs)
    ;; Step 3: Get the return values of the outputs firing
    (map neuron:fire (module:get-output-neurons module))))

;;; Given a list of losses (negatives of gradients) at the output neurons, run a backward pass to get the gradients at the parameters
(define (module:backward! module losses)
  ;; Step 1: Check to make sure the number of losses is correct
  (let ((output-neurons (module:get-output-neurons module)))
    (if (= (length output-neurons) (length losses))
        #t
        (error "Did not supply correct number of losses to the module" module inputs (length module) (length inputs)))
    ;; Step 2: Set the losses for those neurons to the provided losses
    (map neuron:set-raw-loss! output-neurons losses)
    ;; Step 3: Get the gradients at the parameters
    (let ((param-neurons (map param:get-internal-neuron (module:get-params module))))
      (map neuron:grad param-neurons))))

;;; Get a list of the gradients of the module's parameters (in the same order as the module's parameters)
(define (module:get-param-grads module)
  (map param:get-grad (module:get-params module)))
  


#| Ways to combine modules |#

;;; Returns a new module that is a combination of module1 and module2, matching up the outputs
;;; of module1 with the inputs of module2.
;;; NOTE: Don't try joining together multiple things to the same modoule2. It won't work correctly!
(define (module:join2! module1 module2)
  (let ((outputs1 (module:get-output-neurons module1))
        (inputs2 (module:get-input-neurons module2)))
    ;; Error if length mismatch
    (if (= (length outputs1) (length inputs2))
        #t
        (error "Output size of first module not equal to input size of second module" (length outputs1) (length inputs2)))
    (map (lambda (input output) (neuron:join! (list input) output)) outputs1 inputs2)
    (make-module (module:get-input-neurons module1)
                 (module:get-output-neurons module2)
                 (append (module:get-params module1) (module:get-params module2))
                 (lambda () (begin
                              (module:reset! module1)
                              (module:reset! module2))))))

(define (module:join! . modules)
  (if (< 1 (length modules))
        (module:join2! (car modules) (apply module:join! (cdr modules)))
        (car modules)))

;;; Return a new module that is a list of neurons (with no parameters)
(define (module:neuron-list . neurons)
  (make-module neurons
               neurons
               '()
               (lambda () (map neuron:reset! neurons))))


;;; Return a new module that adds together the outputs of all the modules
(define (module:add-right . modules)
  (let ((input-neurons (apply append (map module:get-input-neurons modules)))
        (mid-neurons (apply append (map module:get-output-neurons modules)))
        (params (apply append (map module:get-params modules)))
        (output-neuron (make-add-neuron)))
    (neuron:join! mid-neurons output-neuron)
    (make-module input-neurons
                 (list output-neuron)
                 params
                 (lambda () (begin
                              (neuron:reset! output-neuron)
                              (map module:reset! modules))))))

#| Basic modules |#

;;; Make a basic module that multiplies an input neuron by a single weight
(define (module-single-weight input-neuron)
  (let* ((param (param-uniform-random)) ;; TODO: Allow other random initializations of the parameter
         (weight-neuron (param:get-internal-neuron param))
         (new-neuron (make-mult-neuron)))
    (neuron:join! (list input-neuron weight-neuron) new-neuron)
    (make-module (list input-neuron)
                 (list new-neuron)
                 (list param)
                 (lambda ()
                         (begin
                           (neuron:reset! new-neuron)
                           (neuron:reset! weight-neuron)
                           (neuron:reset! input-neuron))))))

;;; Make a basic module that adds a bias to an input neuron
(define (module-single-bias input-neuron)
  (let* ((param (param-uniform-random)) ;; TODO: Allow other random initializations of the parameter
         (weight-neuron (param:get-internal-neuron param))
         (new-neuron (make-add-neuron)))
    (neuron:join! (list input-neuron weight-neuron) new-neuron)
    (make-module (list input-neuron)
                 (list new-neuron)
                 (list param)
                 (lambda ()
                         (begin
                           (neuron:reset! new-neuron)
                           (neuron:reset! weight-neuron)
                           (neuron:reset! input-neuron))))))

;;; Make a basic module that applies an activation function to an input neuron
(define (module-activation size output-creator)
  (let* ((input-neurons (map (lambda x (make-identity-neuron)) (iota size)))
    (output-neurons (map (lambda x (output-creator)) input-neurons))) ;; TODO: Allow other activation functions
    (map (lambda (i)
    (neuron:join! (list (list-ref input-neurons i)) (list-ref output-neurons i))) (iota (length input-neurons)))
    (make-module input-neurons
                 output-neurons
                 '()
                 (lambda ()
                         (begin
                           (map (lambda (neuron) (neuron:reset! neuron)) input-neurons)
                           (map (lambda (neuron) (neuron:reset! neuron)) output-neurons))))))

;;; Return a module that runs a perceptron on the input neurons
(define (module-perceptron input-neurons)
  (let ((linear-module (apply module:add-right
                             (map module-single-weight input-neurons)))
        (bias (module-single-bias (make-identity-neuron))))
    (module:join! linear-module bias)))

;;; Return a fully connected layer with the corresponding number of inputs and outputs
(define (module-fc num-inputs num-outputs)
  (let* ((input-neurons (map (lambda (x) (make-identity-neuron)) (iota num-inputs)))
         (perceptrons (map (lambda (x) (module-perceptron input-neurons)) (iota num-outputs)))
         (output-neurons (apply append (map module:get-output-neurons perceptrons)))
         (params (apply append (map module:get-params perceptrons))))
    (make-module input-neurons
                 output-neurons
                 params
                 (lambda () (begin
                              (map neuron:reset! input-neurons)
                              (map module:reset! perceptrons))))))
    
                                