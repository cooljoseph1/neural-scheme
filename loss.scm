;;; MSE loss function
(define (loss:mse module target-neurons)
  (let ((predicted-neurons (module:get-output-neurons module))
	(sum-neuron (make-add-neuron))
        (negative-targets (map (lambda x (make-mult-neuron)) target-neurons))
        (diff-neurons (map (lambda x (make-add-neuron)) target-neurons))
        (diff-squared-neurons (map (lambda x (make-pow-neuron 2)) target-neurons)))

    (neuron:join! diff-squared-neurons sum-neuron)
    (letrec ((loop (lambda (i)
		     (if (< i (length predicted-neurons))
			 (begin
			   (neuron:join! (list (list-ref target-neurons i) (make-input-neuron -1)) (list-ref negative-targets i))
			   (neuron:join! (list (list-ref negative-targets i) (list-ref predicted-neurons i)) (list-ref diff-neurons i))
			   (neuron:join! (list (list-ref diff-neurons i)) (list-ref diff-squared-neurons i))
			   (loop (+ i 1)))))))
      (loop 0)
      (make-module (module:get-input-neurons module)
		   (list sum-neuron)
		   (module:get-params module)
		   (lambda () (begin
				(map neuron:reset! (append (list sum-neuron) negative-targets diff-neurons diff-squared-neurons target-neurons))
				(module:reset! module)))))))

;; module outputs are logits, other parameter is one-hot vector representing correct classification
(define (loss:cross-entropy module one-hot)
  (let* ((logits (module:get-output-neurons module))
	 (exps (map (lambda x (make-exp-neuron)) logits))
	 (divs (map (lambda x(make-div-neuron)) logits))
	 (logs (map (lambda x (make-log-neuron)) logits))
	 (exp-sum (make-add-neuron))
	 (one-hot-product (map (lambda x (make-mult-neuron)) logits))
	 (one-hot-sum (make-add-neuron))
	 (neg (make-mult-neuron)))
    (neuron:join! exps exp-sum)
    (neuron:join! (list one-hot-sum (make-input-neuron -1)) neg)
			   (neuron:join! one-hot-product one-hot-sum)
    (letrec ((loop (lambda (i)
		     (if (< i (length logits))
			 (begin
			   (neuron:join! (list (list-ref logits i)) (list-ref exps i))
			   (neuron:join! (list (list-ref exps i) exp-sum) (list-ref divs i))
			   (neuron:join! (list (list-ref divs i)) (list-ref logs i))
			   (neuron:join! (list (list-ref logs i) (list-ref one-hot i)) (list-ref one-hot-product i))
			   (loop (+ i 1))
			   )))))
      (loop 0))
    (make-module (module:get-input-neurons module)
		 (list neg)
		 (module:get-params module)
		 (lambda () (begin
			      (map neuron:reset! (append (list exp-sum one-hot-sum neg) exps divs logs one-hot-product one-hot))
			      (module:reset! module))))))
