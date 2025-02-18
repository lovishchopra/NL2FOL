(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsCasuallyDressed (BoundSet) Bool)
(declare-fun WalkTogether (BoundSet) Bool)
(declare-fun IsWalk (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (and (IsCasuallyDressed a) (WalkTogether a))) (and (forall ((c BoundSet)) (forall ((d BoundSet)) (=> (IsWalk c) (IsCasuallyDressed d)))) (and (forall ((e BoundSet)) (forall ((f BoundSet)) (=> (WalkTogether e) (IsWalk f)))) (forall ((h BoundSet)) (forall ((g BoundSet)) (=> (IsWalk g) (WalkTogether h))))))) (exists ((b BoundSet)) (IsWalk b)))))
(check-sat)
(get-model)