(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsIdeologyFor (BoundSet) Bool)
(declare-fun IsImportantFor (BoundSet) Bool)
(declare-fun IsMoney (BoundSet) Bool)
(declare-fun IsHeadlines (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (IsIdeologyFor b)) (exists ((c BoundSet)) (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsImportantFor b) (or (IsMoney c) (IsHeadlines a)))))))))
(check-sat)
(get-model)