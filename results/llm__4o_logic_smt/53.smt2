(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun CauseCancer (BoundSet) Bool)
(declare-fun ReplaceWith (BoundSet BoundSet) Bool)
(declare-fun VoteFor (BoundSet) Bool)
(declare-fun TearDown (BoundSet) Bool)
(declare-fun On (BoundSet BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (CauseCancer a)) (forall ((g BoundSet)) (forall ((h BoundSet)) (=> (CauseCancer g) (ReplaceWith g h))))) (exists ((e BoundSet)) (exists ((a BoundSet)) (exists ((d BoundSet)) (exists ((c BoundSet)) (and (VoteFor c) (and (TearDown a) (and (ReplaceWith a d) (On d e)))))))))))
(check-sat)
(get-model)