(define (load-data filename)
  (define input-port (open-input-file filename))
  (define data (read input-port))
  (close-input-port input-port)
  data)

(define (write-data filename data)
  (call-with-output-file filename (lambda (f) (write data filename))))