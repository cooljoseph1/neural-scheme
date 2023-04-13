;;;; This module contains tests for different parts of the code to make sure it all works

(load "module.scm")
;; Make a basic fully connected module with 2 inputs and 3 outputs
(define test-module (module-fc 2 3))

;; Test that the forward procedure is working
(pp (module:forward test-module (list 3 2)))

;; Check that we can reset the graph and run another forward pass
(module:reset! test-module)
(pp (module:forward test-module (list 2 3)))

;; Check if we can do a back propagation
(module:backward! test-module (list 1 -1 1))
(pp (module:get-param-grads test-module))

;; Check that if we reset the module and run a forward pass again we get the same thing as earlier
(module:reset! test-module)
(pp (module:forward test-module (list 2 3)))

;; Do back propagation again with the same losses. Check that the gradients have doubled
(module:backward! test-module (list 1 -1 1))
(pp (module:get-param-grads test-module))