(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsWalkingAlong (BoundSet BoundSet) Bool)
(declare-fun IsInRiverStones (BoundSet) Bool)
(declare-fun ( (Bool) Bool)
(declare-fun HasDog (BoundSet) Bool)
(declare-fun IsSleeping (BoundSet) Bool)
(assert (not (=> (and (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((d BoundSet)) (and (exists ((b BoundSet)) (( (and (IsWalkingAlong a b) (IsInRiverStones d)))) (HasDog c))))) (and (forall ((f BoundSet)) (forall ((g BoundSet)) (=> (HasDog f) (IsSleeping g)))) (forall ((i BoundSet)) (forall ((h BoundSet)) (=> (IsSleeping h) (HasDog i)))))) (exists ((e BoundSet)) (IsSleeping e)))))
(check-sat)
(get-model)