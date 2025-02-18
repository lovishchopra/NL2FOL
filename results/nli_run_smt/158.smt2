(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsLittle (BoundSet) Bool)
(declare-fun IsWhite (BoundSet) Bool)
(declare-fun RunsOutside (BoundSet) Bool)
(declare-fun IsOutside (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (and (IsLittle a) (and (IsWhite a) (RunsOutside a)))) (and (forall ((e BoundSet)) (forall ((d BoundSet)) (=> (IsWhite d) (IsOutside e)))) (forall ((f BoundSet)) (forall ((g BoundSet)) (=> (RunsOutside f) (IsOutside g)))))) (exists ((c BoundSet)) (IsOutside c)))))
(check-sat)
(get-model)