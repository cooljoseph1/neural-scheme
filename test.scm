;;;; This module contains tests for different parts of the code to make sure it all works

(load "module.scm")
(define test-module (module-fc 2 3))
(pp (module:forward test-module (list 3 2)))
(module:reset! test-module)
(pp (module:forward test-module (list 2 3)))