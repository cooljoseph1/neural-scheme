(load "load.scm")


(define (train)
  (letrec* ((test-module (module:join! (module-fc 784 20) (module-activation 20 make-relu-neuron) (module-fc 20 10)))
	    (data (load-mnist "mnist_train.csv"))
	    (expected (make-neuron-controllable))
	    (loss-module (loss:mse test-module (list (cadr expected))))
	    (opt (make-adam-optimizer 0.001 loss-module))
	    (loop (lambda (i)
		    (if (< i 100)
			(let* ((sample (list-ref data (random (length data))))
			       (x (car sample))
			       (y (cadr sample)))
			  (module:reset! loss-module)
			  ((car expected) y)
			  (module:forward loss-module x)
			  (if (= (remainder i 1000) 0) (begin ;;(pp (list 'output (module:forward test-module x)))
							      ;;(pp (list 'expected y))
							      ;;(pp (list 'weights (map param:get-weight (module:get-params loss-module))))
							      (pp (list 'loss i (module:forward loss-module x)))))
			  (module:backward! loss-module (list -1))
					;(pp (list "grads" (module:get-param-grads loss-module)))
			  ;;(pp "")
			  (opt)
			  (loop (+ i 1)))
            test-module))))

    (loop 0)))


(define (test module)
  (letrec* (
    (data (load-mnist "mnist_test.csv"))
	(expected (make-neuron-controllable))
    (loss-module (loss:mse test-module (list (cadr expected))))
    (loop (lambda (i corr)
		    (if (< i 100)
			(let* ((sample (list-ref data (random (length data))))
			       (x (car sample))
			       (y (cadr sample)))
			  (module:reset! loss-module)
			  ((car expected) y)
			  (module:forward loss-module x)
			  (begin (pp (list 'predicted (max-idx (module:forward test-module x))))
							      (pp (list 'expected (max-idx y)))
							      ;;(pp (list 'weights (map param:get-weight (module:get-params loss-module))))
							      (pp (list 'loss i (module:forward loss-module x))))
					;(pp (list "grads" (module:get-param-grads loss-module)))
			  ;;(pp "")
			  (loop (+ i 1) (if (eqv? (max-idx (module:forward test-module x)) (max-idx y))  (+ corr 1) corr )))
            test-module)
            corr)))

    (loop 0)))

(define module (train))

(pp (list '%correct: (/ (test module) 100)))