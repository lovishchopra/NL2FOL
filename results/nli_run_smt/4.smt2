(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsWearing (BoundSet BoundSet) Bool)
(declare-fun IsPublicSpeaker (BoundSet) Bool)
(declare-fun IsInPlaidShirt (BoundSet) Bool)
(declare-fun IsSpeakingOnMicrophone (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((b BoundSet)) (IsWearing a b))) (exists ((e BoundSet)) (exists ((c BoundSet)) (exists ((d BoundSet)) (and (IsPublicSpeaker c) (and (IsInPlaidShirt d) (IsSpeakingOnMicrophone e)))))))))
(check-sat)
(get-model)