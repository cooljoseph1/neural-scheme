(define (parse-csv-file filename)
  (let ((lines (file->lines filename)))
    (map (lambda (line) (string-split line ","))
         lines)))

(define (file->lines filename)
  (let ((port (open-input-file filename)))
    (let loop ((lines '()))
      (let ((line (read-line port)))
        (if (eof-object? line)
            (begin
              (close-input-port port)
              (reverse lines))
            (loop (cons line lines)))))))

(define (string-split str sep)
  (let loop ((str str) (tokens '()) (token '()))
    (cond ((string-null? str) (reverse (cons (list->string (reverse token)) tokens)))
          ((char=? (string-ref str 0) (string-ref sep 0))
           (loop (substring str 1) (cons (list->string (reverse token)) tokens) '()))
          (else (loop (substring str 1) tokens (cons (string-ref str 0) token))))))


(define (mnist-one-hot-encode index)
    (map (lambda (i) (if (= i index) 1 0)) (iota 10)))

(define (max-idx vals)
  (list-index (apply max vals) vals))

(define (load-mnist filename)
  (let ((raw-data (cdr (parse-csv-file filename))))
    (map (lambda (line) (list (map string->number (cdr line)) 
                        (mnist-one-hot-encode (string->number (car line)))))
          raw-data)))