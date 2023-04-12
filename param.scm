(load "neuron.scm")

(define (param initial-weight)
  (let* ((weight initial-weight)
         (gradient 0)

         (get-weight (lambda () weight))
         (set-weight! (lambda (new-weight) (set! weight new-weight)))

         (get-grad (lambda () gradient))
         (set-grad! (lambda (new-grad) (set! gradient new-grad)))
         (zero-grad! (lambda () (set! gradient 0)))

         (internal-neuron (make-neuron get-weight
                                       (lambda (inputs output grad)
                                               (begin
                                                 (set-grad! (+ (get-grad) grad))
                                                 '())))))
    (vector get-weight set-weight! get-grad set-grad! zero-grad! internal-neuron)))

(define (param:get-weight param)
  ((vector-ref param 0)))

(define (param:set-weight! param new-weight)
  ((vector-ref param 1) new-weight))

(define (param:get-grad param)
  ((vector-ref param 2)))

(define (param:set-grad! param new-grad)
  ((vector-ref param 3) new-grad))

(define (param:zero-grad! param)
  ((vector-ref param 4)))

(define (param:get-internal-neuron param)
  (vector-ref param 5))

;;; Return a parameter with a random weight chosen uniformly at random between lower and upper
(define (param-uniform-random . args)
  (let* ((lower (if (null? args) -100 (car args)))
         (upper (if (null? args) -99 (if (null? (cdr args)) -99 (cadr args))))
         (weight (+ lower (* (random) (- upper lower)))))
    (param weight)))