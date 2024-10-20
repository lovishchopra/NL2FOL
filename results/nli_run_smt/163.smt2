(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsBrownAndWhite (BoundSet) Bool)
(declare-fun IsNear (BoundSet BoundSet) Bool)
(declare-fun IsChainedTo (BoundSet BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsBrownAndWhite a) (IsNear a b)))) (forall ((f BoundSet)) (forall ((e BoundSet)) (forall ((h BoundSet)) (forall ((g BoundSet)) (=> (IsChainedTo e f) (IsNear g h))))))) (exists ((c BoundSet)) (exists ((d BoundSet)) (IsChainedTo c d))))))
(check-sat)
(get-model)