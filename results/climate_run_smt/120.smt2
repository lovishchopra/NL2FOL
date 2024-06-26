(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsWoman (BoundSet) Bool)
(declare-fun IsDarkHair (BoundSet) Bool)
(declare-fun IsDarkHeaded (BoundSet) Bool)
(declare-fun IsBending (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsWoman a) (IsDarkHair b)))) (and (forall ((f BoundSet)) (forall ((g BoundSet)) (=> (IsDarkHair f) (IsDarkHeaded g)))) (forall ((h BoundSet)) (forall ((i BoundSet)) (=> (IsDarkHeaded h) (IsDarkHair i)))))) (exists ((a BoundSet)) (exists ((d BoundSet)) (and (IsBending a) (IsDarkHeaded d)))))))
(check-sat)
(get-model)