(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsInBlueCollaredShirt (BoundSet) Bool)
(declare-fun IsSittingInFrontOf (BoundSet BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsInBlueCollaredShirt b) (IsSittingInFrontOf b a)))) (forall ((e BoundSet)) (forall ((g BoundSet)) (forall ((f BoundSet)) (=> (IsSittingInFrontOf e f) (IsInBlueCollaredShirt g)))))) (exists ((a BoundSet)) (exists ((d BoundSet)) (IsSittingInFrontOf d a))))))
(check-sat)
(get-model)