;;;; This module contains tests for different parts of the code to make sure it all works

(load "module.scm")
(define test-module (module-fc 2 3))
(module:forward test-module (list 3 2))