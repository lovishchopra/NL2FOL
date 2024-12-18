(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsUsingHandsFreeMicrophone (BoundSet) Bool)
(declare-fun IsWearingBlueAndYellowBowlerHat (BoundSet) Bool)
(declare-fun IsSpeakingIntoHeadsetMicrophone (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (IsUsingHandsFreeMicrophone a)) (exists ((a BoundSet)) (and (IsWearingBlueAndYellowBowlerHat a) (IsSpeakingIntoHeadsetMicrophone a))))))
(check-sat)
(get-model)