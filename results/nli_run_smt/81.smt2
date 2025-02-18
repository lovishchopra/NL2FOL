(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsCurlyHaired (BoundSet) Bool)
(declare-fun IsDrinkingJuice (BoundSet) Bool)
(declare-fun IsTakingABigBite (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (and (IsCurlyHaired a) (IsDrinkingJuice a))) (exists ((a BoundSet)) (and (IsTakingABigBite a) (IsCurlyHaired a))))))
(check-sat)
(get-model)