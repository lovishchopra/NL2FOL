(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsInBlueWrapping (BoundSet) Bool)
(declare-fun IsAboutToKick (BoundSet) Bool)
(declare-fun IsInRedCloth (BoundSet) Bool)
(declare-fun IsDrinksMilk (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (and (IsInBlueWrapping a) (or (IsAboutToKick a) (IsInRedCloth a)))) (exists ((b BoundSet)) (IsDrinksMilk b)))))
(check-sat)
(get-model)