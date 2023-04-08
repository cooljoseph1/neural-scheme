;;;; Modules
;;; Modules are the basic building blocks of a neural network. They take in inputs
;;; and give outputs through a forward pass. They can also take in gradients at their
;;; output neurons and give out gradients at their input neurons in their backward pass,
;;; saving internal gradients along the way.

;;; First, load in common utils:
(load "utils.scm")


;;; Create a module given a procedure that implements the forward pass and
;;; a procedure that implements the backward pass.
;;; TODO: Compile an automatic backward procedure out of the forward one
;;; TODO 2: Make a module save information along the way as it goes forward
;;;         so the backward pass can be done correctly
(define (make-module forward backward)
  (cons forward backward))

;;; Return the forward procedure of a module
(define (get-forward module)
  (car module))

;;; Return the backward procedure of a module
(define (get-backward module)
  (cadr module))

;;; Call the forward pass of a module
(define (forward module inputs)
  (apply (get-forward module) inputs))

;;; Call the backward pass of a module
(define (backward module gradients)
  (apply (get-backward module) gradients))

;;; Combine two modules together (this is where things get interesting)
;;; Note: In the forward pass module1 is called before module2.
(define (compose-modules module2 module1)
  (make-module (compose (get-forward module2) (get-forward module1))       ; Forward pass
               (compose (get-backward module1) (get-backward module2))))   ; Backward pass