(define (make-sgd-optimizer lr module)
  (let ((params (module:get-params module)))
    (lambda ()
      (letrec ((loop (lambda (i)
		       (if (< i (length params))
			   (let ((param (list-ref params i)))
			     (param:set-weight! param (- (param:get-weight param) (* lr (param:get-grad param))))
			     (loop (+ i 1)))))))
	(loop 0)))))

(define (set-list-idx! l idx val)
  (if (= idx 0)
      (set-car! l val)
      (set-list-idx! (cdr l) (- idx 1) val)))
(define (make-adam-optimizer lr module)
  (let* ((params (list->vector (module:get-params module)))
	 (t 1)
	 (m (vector-map (lambda x 0) params))
	 (v (vector-map (lambda x 0) params))
	 (beta1 0.9)
	 (beta2 0.999)
	 (eps 1e-8))
    (lambda ()
      (map (lambda (i)
	     (let ((param (vector-ref params i)))
	       (vector-set! m i (+ (* beta1 (vector-ref m i)) (* (- 1 beta1) (param:get-grad param))))
	       (vector-set! v i (+ (* beta2 (vector-ref v i)) (* (- 1 beta2) (expt (param:get-grad param) 2))))
	       (let ((mhat (/ (vector-ref m i) (- 1 (expt beta1 t))))
		     (vhat (/ (vector-ref v i) (- 1 (expt beta2 t)))))
		 (param:set-weight!
		  param
		  (-
		   (param:get-weight param)
		   (* lr (/ mhat (+ (sqrt vhat) eps))))))))
	   (iota (vector-length params)))
      (set! t (+ t 1)))))
