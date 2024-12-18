(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsTakingCareOf (BoundSet BoundSet) Bool)
(declare-fun IsNear (BoundSet BoundSet) Bool)
(declare-fun IsMadePrimarilyOfStone (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((b BoundSet)) (IsTakingCareOf a b))) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsNear a c) (IsMadePrimarilyOfStone c)))))))
(check-sat)
(get-model)