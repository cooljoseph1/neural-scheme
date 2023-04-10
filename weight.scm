(define (param initial-weight)
  (let* ((weight initial-weight)
         (set-weight (lambda (new-weight) (set! weight new-weight)))
         (get-weight (lambda () weight))
         (forward (lambda () weight))))
    (cons get-weight set-weight))

(define (param:get-weight param)
  ((car param)))

(define (param:set-weight param new-weight)
  ((cdr param) new-weight))