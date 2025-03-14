(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsSkateboarding (BoundSet) Bool)
(declare-fun IsWatching (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (IsSkateboarding a)) (exists ((b BoundSet)) (IsWatching b)))))
(check-sat)
(get-model)