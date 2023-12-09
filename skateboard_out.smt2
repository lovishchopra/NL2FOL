sat
(model
; cardinality of BoundSet is 2
(declare-sort BoundSet 0)
; rep: @uc_BoundSet_0
; rep: @uc_BoundSet_1
; cardinality of UnboundSet is 1
(declare-sort UnboundSet 0)
; rep: @uc_UnboundSet_0
(define-fun cheese () UnboundSet @uc_UnboundSet_0)
(define-fun T ((BOUND_VARIABLE_369 BoundSet)) Bool true)
(define-fun L ((BOUND_VARIABLE_381 BoundSet) (BOUND_VARIABLE_382 UnboundSet)) Bool (ite (= @uc_BoundSet_0 BOUND_VARIABLE_381) (= @uc_UnboundSet_0 BOUND_VARIABLE_382) false))
)
