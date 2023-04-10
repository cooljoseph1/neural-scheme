;;;; These are common useful functions that don't exist in native MIT Scheme.

;;; Replace the values of a list with the new values
(define (replace-list! lst new-values)
  (cond ((null? lst) '())
         (else (set-car! lst (car new-values))
               (replace-list! (cdr lst) (cdr new-values)))))

;;; Get a list (0 1 ... n-1) for a list that is of length n
(define (get-indexes n)
  (let loop ((idx (- n 1)) (indexes '()))
    (cond ((< idx 0) indexes)
          (else (loop (- idx 1)
                      (cons idx indexes))))))

;;; Compose two functions together
(define (compose f g)
  (lambda args
    (call-with-values (lambda () (apply g args))
                      (lambda result (apply f result)))))

;;; Compose two functions together, but f and g return lists instead of values
(define (list-compose f g)
  (lambda args
    (apply f (apply g args))))