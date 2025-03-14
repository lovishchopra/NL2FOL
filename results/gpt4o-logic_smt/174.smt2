(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsInRedCoat (BoundSet) Bool)
(declare-fun IsInBlueHeadWrap (BoundSet) Bool)
(declare-fun IsInJeans (BoundSet) Bool)
(declare-fun IsMakingSnowAngel (BoundSet) Bool)
(declare-fun IsOutside (BoundSet) Bool)
(declare-fun PlaysInSnow (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (and (IsInRedCoat a) (and (IsInBlueHeadWrap a) (and (IsInJeans a) (IsMakingSnowAngel a))))) (and (forall ((g BoundSet)) (=> (IsInBlueHeadWrap g) (IsOutside g))) (and (forall ((h BoundSet)) (=> (IsInBlueHeadWrap h) (PlaysInSnow h))) (and (forall ((i BoundSet)) (=> (IsMakingSnowAngel i) (IsOutside i))) (and (forall ((j BoundSet)) (=> (IsOutside j) (IsMakingSnowAngel j))) (forall ((k BoundSet)) (=> (IsMakingSnowAngel k) (PlaysInSnow k)))))))) (exists ((a BoundSet)) (and (IsOutside a) (PlaysInSnow a))))))
(check-sat)
(get-model)