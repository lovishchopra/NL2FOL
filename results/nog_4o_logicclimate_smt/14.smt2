(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsTan (BoundSet) Bool)
(declare-fun HasWoolHat (BoundSet) Bool)
(declare-fun IsRunning (BoundSet) Bool)
(declare-fun IsLeaningOver (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsTan b) (and (HasWoolHat b) (and (IsRunning b) (IsLeaningOver b a)))))) (exists ((a BoundSet)) (exists ((c BoundSet)) (and (IsTan c) (and (IsRunning c) (IsLeaningOver c a))))))))
(check-sat)
(get-model)