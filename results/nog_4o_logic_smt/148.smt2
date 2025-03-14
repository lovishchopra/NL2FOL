(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsGroup (BoundSet) Bool)
(declare-fun SingsTogether (BoundSet) Bool)
(declare-fun IsWearing (BoundSet BoundSet) Bool)
(declare-fun IsSinging (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsGroup a) (and (SingsTogether a) (IsWearing b a))))) (exists ((c BoundSet)) (IsSinging c)))))
(check-sat)
(get-model)