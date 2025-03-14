(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsVotingFor (BoundSet BoundSet) Bool)
(declare-fun IsCandidateForPresident (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (exists ((c BoundSet)) (and (IsVotingFor c a) (IsCandidateForPresident a b))))) (exists ((b BoundSet)) (exists ((d BoundSet)) (exists ((a BoundSet)) (and (IsVotingFor d a) (IsCandidateForPresident a b))))))))
(check-sat)
(get-model)