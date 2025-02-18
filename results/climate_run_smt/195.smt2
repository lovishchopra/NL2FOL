(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsLookingThrough (BoundSet) Bool)
(declare-fun IsTelescope (BoundSet) Bool)
(declare-fun IsUsed (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsLookingThrough a) (IsTelescope b)))) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsUsed a) (IsTelescope b)))))))
(check-sat)
(get-model)