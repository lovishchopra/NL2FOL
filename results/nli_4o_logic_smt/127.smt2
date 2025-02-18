(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsBlond (BoundSet) Bool)
(declare-fun HasTattoo (BoundSet) Bool)
(declare-fun IsTattooOf (BoundSet BoundSet) Bool)
(declare-fun IsOn (BoundSet BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (exists ((d BoundSet)) (exists ((c BoundSet)) (and (IsBlond b) (and (HasTattoo b) (and (IsTattooOf a c) (IsOn a d)))))))) (forall ((i BoundSet)) (forall ((g BoundSet)) (forall ((h BoundSet)) (=> (IsOn g h) (HasTattoo i)))))) (exists ((e BoundSet)) (HasTattoo e)))))
(check-sat)
(get-model)