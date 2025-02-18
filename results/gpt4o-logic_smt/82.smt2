(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsInGreatShape (BoundSet) Bool)
(declare-fun AreWorthTheMoney (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (IsInGreatShape a)) (exists ((a BoundSet)) (AreWorthTheMoney a)))))
(check-sat)
(get-model)