(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun SingTogether (BoundSet) Bool)
(declare-fun WearMatchingBlackDresses (BoundSet) Bool)
(declare-fun AreWomen (BoundSet) Bool)
(declare-fun AreSinging (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (and (SingTogether a) (WearMatchingBlackDresses a))) (and (forall ((d BoundSet)) (forall ((e BoundSet)) (=> (SingTogether d) (AreWomen e)))) (and (forall ((g BoundSet)) (forall ((f BoundSet)) (=> (AreWomen f) (SingTogether g)))) (and (forall ((i BoundSet)) (forall ((h BoundSet)) (=> (SingTogether h) (AreSinging i)))) (and (forall ((j BoundSet)) (forall ((k BoundSet)) (=> (AreSinging j) (SingTogether k)))) (forall ((m BoundSet)) (forall ((l BoundSet)) (=> (WearMatchingBlackDresses l) (AreSinging m))))))))) (exists ((c BoundSet)) (and (AreWomen c) (AreSinging c))))))
(check-sat)
(get-model)