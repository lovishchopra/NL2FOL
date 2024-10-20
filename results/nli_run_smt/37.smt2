(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsSweet (BoundSet) Bool)
(declare-fun IsOnIceSkates (BoundSet) Bool)
(declare-fun IsPerformingOnTheIce (BoundSet) Bool)
(declare-fun IsFemale (BoundSet) Bool)
(declare-fun IsInPool (BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsSweet a) (and (IsOnIceSkates b) (IsPerformingOnTheIce c)))))) (exists ((e BoundSet)) (exists ((d BoundSet)) (and (IsFemale d) (IsInPool e)))))))
(check-sat)
(get-model)