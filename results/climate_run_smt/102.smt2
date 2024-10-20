(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsSat (BoundSet) Bool)
(declare-fun IsPasserby (BoundSet) Bool)
(declare-fun IsInDoorFrame (BoundSet) Bool)
(declare-fun IsOnGraySidewalk (BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsSat a) (= (or (IsPasserby c) a) b))))) (exists ((e BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsSat a) (and (IsInDoorFrame b) (IsOnGraySidewalk e)))))))))
(check-sat)
(get-model)