(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsWearingBlackCap (BoundSet) Bool)
(declare-fun IsSuspendedInAir (BoundSet) Bool)
(declare-fun IsOnSwing (BoundSet) Bool)
(declare-fun IsSwingingWithChildren (BoundSet) Bool)
(declare-fun IsInBlackHat (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (and (IsWearingBlackCap a) (and (IsSuspendedInAir a) (IsOnSwing a)))) (forall ((g BoundSet)) (forall ((f BoundSet)) (=> (IsOnSwing f) (IsSwingingWithChildren g))))) (exists ((d BoundSet)) (and (IsInBlackHat d) (IsSwingingWithChildren d))))))
(check-sat)
(get-model)