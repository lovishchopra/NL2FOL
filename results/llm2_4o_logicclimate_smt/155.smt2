(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsCampingOn (BoundSet BoundSet) Bool)
(declare-fun IsOutdoors (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (IsCampingOn a b))) (and (forall ((d BoundSet)) (forall ((f BoundSet)) (forall ((e BoundSet)) (=> (IsCampingOn d e) (IsOutdoors f))))) (forall ((g BoundSet)) (forall ((h BoundSet)) (forall ((i BoundSet)) (=> (IsOutdoors g) (IsCampingOn h i))))))) (exists ((c BoundSet)) (IsOutdoors c)))))
(check-sat)
(get-model)