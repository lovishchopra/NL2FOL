(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun Overpowers (BoundSet BoundSet) Bool)
(declare-fun IsFrom (BoundSet BoundSet) Bool)
(declare-fun LeadsTo (BoundSet BoundSet) Bool)
(declare-fun IsNetLandAreaGain (BoundSet) Bool)
(assert (not (=> (and (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((d BoundSet)) (exists ((b BoundSet)) (and (Overpowers b c) (and (IsFrom d c) (LeadsTo b a))))))) (and (forall ((f BoundSet)) (forall ((g BoundSet)) (forall ((h BoundSet)) (=> (Overpowers f g) (IsNetLandAreaGain h))))) (forall ((i BoundSet)) (forall ((j BoundSet)) (forall ((k BoundSet)) (=> (LeadsTo i j) (IsNetLandAreaGain k))))))) (exists ((a BoundSet)) (IsNetLandAreaGain a)))))
(check-sat)
(get-model)