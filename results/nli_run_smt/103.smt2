(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsWearingBlackTop (BoundSet) Bool)
(declare-fun IsSpeakingIntoBlueTippedMicrophone (BoundSet) Bool)
(declare-fun ( (Bool) Bool)
(declare-fun IsBesideHer (BoundSet) Bool)
(declare-fun IsSpeakingIntoMicrophone (BoundSet) Bool)
(assert (not (=> (and (exists ((e BoundSet)) (exists ((c BoundSet)) (and (exists ((a BoundSet)) (( (and (IsWearingBlackTop c) (IsSpeakingIntoBlueTippedMicrophone e)))) (IsBesideHer a)))) (forall ((j BoundSet)) (forall ((k BoundSet)) (=> (IsSpeakingIntoBlueTippedMicrophone j) (IsSpeakingIntoMicrophone k))))) (exists ((h BoundSet)) (exists ((g BoundSet)) (and (IsSpeakingIntoMicrophone g) (IsBesideHer h)))))))
(check-sat)
(get-model)