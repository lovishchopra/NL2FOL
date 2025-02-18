(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsSmall (BoundSet) Bool)
(declare-fun IsOnPavedPath (BoundSet) Bool)
(declare-fun IsA (BoundSet) Bool)
(declare-fun IsConsequence (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsSmall a) (IsOnPavedPath b)))) (exists ((c BoundSet)) (exists ((d BoundSet)) (and (IsA c) (IsConsequence d)))))))
(check-sat)
(get-model)