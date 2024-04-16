(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsDoingHandstand (BoundSet) Bool)
(declare-fun IsLucky (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (IsDoingHandstand a)) (exists ((c BoundSet)) (IsLucky c)))))
(check-sat)
(get-model)