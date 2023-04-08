;;;; These are common useful functions that don't exist in native MIT Scheme.

;;; Compose two functions together
(define (compose f g)
  (lambda args
    (call-with-values (lambda () (apply g args))
                      (lambda result (apply f result)))))