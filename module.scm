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
(define (module:get-parameters module)
  (caddr module))

;;; Return the reset function of the module
(define (module:get-reset-function module)
  (cadddr module))

;;; Reset the module to prepare for a forward pass
(define (module:reset! module)
  ((module:get-reset-function module)))

;;; Given a list of inputs, run it through the module. (Note: This is mostly only so complicated because we have to attach temporary inputs)
(define (module:forward module inputs)
  ;; Step 1: Create temporary inputs and attach them to the input neurons
  (let ((input-neurons (module:get-input-neurons module)))
    (if (= (length input-neurons) (length inputs))
        #t
        (error "Did not supply correct number of inputs to a module" module inputs (length module) (length inputs)))
    (let* ((old-inputs (map module:get-input-neurons input-neurons))
           (temp-inputs (map make-input-neuron inputs))
           (temp-module (module:neuron-list temp-inputs)))
      (module:join! temp-module module)
      ;; Step 2: Get the outputs of the forward pass
      (let ((outputs (map neuron:fire (module:get-output-neurons module))))
        ;; Step 3: Get rid of the temporary inputs
        (map neuron:set-input-neurons! input-neurons old-inputs)
        ;; Step 4: Return the outputs
        outputs))))

;;; Given a list of gradients at the output neuron, run a backward pass to get the gradients at the parameters
;;; (define (module:backward module gradients)
  


#| Ways to combine modules |#

;;; Join the outputs of module1 with the inputs of module2, matching them up one by one
(define (module:join! module1 module2)
  (let ((outputs1 (module:get-output-neurons module1))
        (inputs2 (module:get-input-neurons module2)))
    ;; Error if length mismatch
    (if (= (length outputs1) (length inputs2))
        #t
        (error "Output size of first module not equal to input size of second module" (length outputs1) (length inputs2)))
    (map (lambda (input output) (neuron:join! (list input) output)) outputs1 inputs2)
    (make-module (module:get-input-neurons module1)
                 (module:get-output-neurons module2)
                 (append (module:get-parameters module1) (module:get-parameters module2))
                 (lambda () (begin
                              (module:reset! module1)
                              (module:reset! module2))))))

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

;;; Make a basic module that applies an activation function to an input neuron
(define (module-activation input-neuron)
  (let* ((output-neuron (make-relu-neuron))) ;; TODO: Allow other activation functions
    (neuron:join! (list input-neuron) output-neuron)
    (make-module (list input-neuron)
                 (list output-neuron)
                 '()
                 (lambda ()
                         (begin
                           (neuron:reset! output-neuron)
                           (neuron:reset! input-neuron))))))

;;; Return a module that runs a perceptron on the input neurons
(define (module-perceptron input-neurons)
  (let ((linear-module (apply module:add-right
                             (map module-single-weight input-neurons)))
        (act-module (module-activation (make-identity-neuron))))
    (module:join! linear-module act-module)))

;;; Return a fully connected layer with the corresponding number of inputs and outputs
(define (module-fc num-inputs num-outputs)
  (let* ((input-neurons (map (lambda (x) (make-identity-neuron)) (iota num-inputs)))
         (perceptrons (map (lambda (x) (module-perceptron input-neurons))) (iota num-outputs))
         (output-neurons (apply append (map module:get-output-neurons perceptrons)))
         (params (apply append (map module:get-params modules))))
    (make-module input-neurons
                 output-neurons
                 params
                 (lambda () (begin
                              (map neuron:reset! input-neruons)
                              (map module:reset! perceptrons))))))
    
                                