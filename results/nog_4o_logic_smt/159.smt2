(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsBearded (BoundSet) Bool)
(declare-fun SitsNear (BoundSet BoundSet) Bool)
(declare-fun IsWhite (BoundSet) Bool)
(declare-fun IsSitting (BoundSet) Bool)
(declare-fun SitsOutside (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsBearded a) (and (SitsNear a b) (IsWhite b))))) (exists ((d BoundSet)) (exists ((c BoundSet)) (and (IsSitting c) (and (SitsOutside c) (SitsNear c d))))))))
(check-sat)
(get-model)