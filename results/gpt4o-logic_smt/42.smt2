(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsPreyedOn (BoundSet) Bool)
(declare-fun IsSlipperySlope (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsPreyedOn a) (IsPreyedOn b)))) (exists ((c BoundSet)) (IsSlipperySlope c)))))
(check-sat)
(get-model)