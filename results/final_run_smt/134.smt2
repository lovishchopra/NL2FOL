(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsPlayingMiniGolf (BoundSet BoundSet) Bool)
(declare-fun IsUsingTennisBalls (BoundSet) Bool)
(declare-fun ( (Bool) Bool)
(declare-fun IsUsingEnlargedGolfPutters (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (IsPlayingMiniGolf a b))) (exists ((g BoundSet)) (exists ((d BoundSet)) (exists ((e BoundSet)) (or (exists ((f BoundSet)) (( (and (IsPlayingMiniGolf d e) (IsUsingTennisBalls f)))) (and (IsPlayingMiniGolf d e) (IsUsingEnlargedGolfPutters g)))))))))
(check-sat)
(get-model)