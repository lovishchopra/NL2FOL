(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsWalkingDownHill (BoundSet) Bool)
(declare-fun IsOnKnees (BoundSet) Bool)
(declare-fun IsOutside (BoundSet) Bool)
(assert (not (=> (and (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsWalkingDownHill a) (IsOnKnees c)))) (and (forall ((h BoundSet)) (forall ((g BoundSet)) (=> (IsWalkingDownHill g) (IsOutside h)))) (and (forall ((i BoundSet)) (forall ((j BoundSet)) (=> (IsOnKnees i) (IsOutside j)))) (forall ((k BoundSet)) (forall ((l BoundSet)) (=> (IsOutside k) (IsOnKnees l))))))) (exists ((e BoundSet)) (IsOutside e)))))
(check-sat)
(get-model)