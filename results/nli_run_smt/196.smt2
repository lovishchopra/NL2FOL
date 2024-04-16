(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsReachingFor (BoundSet BoundSet) Bool)
(declare-fun IsTalking (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((b BoundSet)) (IsReachingFor a b))) (exists ((d BoundSet)) (IsTalking d)))))
(check-sat)
(get-model)