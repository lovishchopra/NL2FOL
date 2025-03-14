(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsSittingOn (BoundSet BoundSet) Bool)
(declare-fun IsInFrontOf (BoundSet BoundSet) Bool)
(declare-fun IsSeatedOn (BoundSet BoundSet) Bool)
(declare-fun IsOutside (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((c BoundSet)) (exists ((b BoundSet)) (and (IsSittingOn a b) (IsInFrontOf b c))))) (and (forall ((e BoundSet)) (forall ((d BoundSet)) (=> (IsSittingOn d e) (IsSeatedOn d e)))) (forall ((f BoundSet)) (forall ((g BoundSet)) (=> (IsSeatedOn f g) (IsSittingOn f g)))))) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsSeatedOn a b) (IsOutside b)))))))
(check-sat)
(get-model)