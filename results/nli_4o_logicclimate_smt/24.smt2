(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun Increased (BoundSet) Bool)
(declare-fun AreRising (BoundSet) Bool)
(declare-fun HaveShownNoIncrease (BoundSet) Bool)
(declare-fun IsNotExperiencing (BoundSet BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((c BoundSet)) (exists ((b BoundSet)) (and (Increased a) (and (AreRising b) (HaveShownNoIncrease c)))))) (forall ((h BoundSet)) (forall ((f BoundSet)) (forall ((g BoundSet)) (=> (IsNotExperiencing f g) (HaveShownNoIncrease h)))))) (exists ((e BoundSet)) (exists ((d BoundSet)) (IsNotExperiencing d e))))))
(check-sat)
(get-model)