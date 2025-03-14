(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsLousyRoleModel (BoundSet) Bool)
(declare-fun IsNotGoodRoleModel (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (IsLousyRoleModel a)) (exists ((a BoundSet)) (IsNotGoodRoleModel a)))))
(check-sat)
(get-model)