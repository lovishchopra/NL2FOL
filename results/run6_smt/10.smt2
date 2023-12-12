(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsCaught (BoundSet BoundSet) Bool)
(declare-fun IsPlaying (BoundSet) Bool)
(declare-fun Threw (BoundSet BoundSet) Bool)
(declare-fun IsTheSamePerson (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((b BoundSet)) (IsCaught a b))) (exists ((d BoundSet)) (exists ((e BoundSet)) (exists ((c BoundSet)) (and (IsPlaying c) (and (Threw d e) (IsTheSamePerson d e)))))))))
(check-sat)
(get-model)