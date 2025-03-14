(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsScared (BoundSet) Bool)
(declare-fun MakesScared (BoundSet BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (IsScared a)) (forall ((e BoundSet)) (forall ((f BoundSet)) (forall ((d BoundSet)) (=> (IsScared d) (MakesScared e f)))))) (exists ((a BoundSet)) (exists ((b BoundSet)) (MakesScared b a))))))
(check-sat)
(get-model)