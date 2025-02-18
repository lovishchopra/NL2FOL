(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsWaiting (BoundSet) Bool)
(declare-fun IsToUse (BoundSet) Bool)
(declare-fun IsInField (BoundSet) Bool)
(declare-fun IsOutside (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsWaiting a) (and (IsToUse b) (IsInField b))))) (and (forall ((f BoundSet)) (forall ((e BoundSet)) (=> (IsWaiting e) (IsOutside f)))) (and (forall ((g BoundSet)) (forall ((h BoundSet)) (=> (IsOutside g) (IsWaiting h)))) (and (forall ((i BoundSet)) (forall ((j BoundSet)) (=> (IsOutside i) (IsToUse j)))) (forall ((l BoundSet)) (forall ((k BoundSet)) (=> (IsOutside k) (IsInField l)))))))) (exists ((d BoundSet)) (IsOutside d)))))
(check-sat)
(get-model)