(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsExports (BoundSet) Bool)
(declare-fun IsRedMeat (BoundSet) Bool)
(declare-fun HasAbscesses (BoundSet) Bool)
(declare-fun IsInMeat (BoundSet) Bool)
(declare-fun HasCollectionsOfPus (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsExports a) (IsRedMeat b)))) (exists ((e BoundSet)) (exists ((a BoundSet)) (exists ((d BoundSet)) (and (HasAbscesses a) (and (IsInMeat e) (HasCollectionsOfPus d)))))))))
(check-sat)
(get-model)