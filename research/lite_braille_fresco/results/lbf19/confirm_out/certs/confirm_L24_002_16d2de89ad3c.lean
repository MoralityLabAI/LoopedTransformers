namespace LBF19Cert

theorem c_sym_prov_readout_107_B512 (A : Type) (f_a f_b f_c f_d f_e f_f : A -> A) (x : A) (h0 : forall z : A, f_c (f_f z) = f_b (f_f z)) (h1 : forall z : A, f_b (f_e z) = f_a (f_d z)) (h2 : forall z : A, f_d (f_e (f_e z)) = f_d (f_f (f_a z))) (h3 : forall z : A, f_e (f_f z) = f_e (f_e z)) (h4 : forall z : A, f_d (f_c z) = f_a (f_b z)) (h5 : forall z : A, f_e (f_c (f_c z)) = f_e (f_b (f_d z))) (h6 : forall z : A, f_b (f_b (f_c z)) = f_f (f_f (f_e z))) (h7 : forall z : A, f_a (f_e z) = f_f (f_c z)) (h8 : forall z : A, f_f (f_a z) = f_f (f_e z)) (h9 : forall z : A, f_d (f_e (f_d z)) = f_e (f_f (f_b z))) (h10 : forall z : A, f_a (f_a z) = f_c (f_d z)) (h11 : forall z : A, f_d (f_b z) = f_f (f_f z)) : f_d (f_b (f_f (f_c (f_e (f_f (f_a (f_c (f_e (f_d (f_d (f_b (f_a (f_b (f_c (f_b (f_d (f_b (f_c (f_d (f_e (f_d (f_f (f_e x))))))))))))))))))))))) = f_f (f_d (f_b (f_b (f_f (f_f (f_e (f_c (f_e (f_d (f_f (f_f (f_a (f_b (f_c (f_c (f_d (f_b (f_c (f_d (f_b (f_b (f_f (f_a x))))))))))))))))))))))) := by
  conv => lhs; rw [<- h7 (f_e (f_f (f_a (f_c (f_e (f_d (f_d (f_b (f_a (f_b (f_c (f_b (f_d (f_b (f_c (f_d (f_e (f_d (f_f (f_e x))))))))))))))))))))]
  conv => lhs; rw [h11 (f_c (f_d (f_e (f_d (f_f (f_e x))))))]
  conv => lhs; rw [h9 (f_f (f_e x))]
  conv => lhs; rw [<- h3 (f_f (f_a (f_c (f_e (f_d (f_d (f_b (f_a (f_b (f_c (f_b (f_f (f_f (f_c (f_e (f_f (f_b (f_f (f_e x)))))))))))))))))))]
  conv => lhs; rw [<- h7 (f_e (f_f (f_b (f_f (f_e x)))))]
  conv => lhs; rw [<- h3 (f_f (f_b (f_f (f_e x))))]
  conv => lhs; rw [<- h0 (f_a (f_e (f_f (f_f (f_b (f_f (f_e x)))))))]
  conv => lhs; rw [<- h8 x]
  conv => lhs; rw [h7 (f_f (f_f (f_b (f_f (f_a x)))))]
  conv => lhs; rw [h8 (f_c (f_e (f_d (f_d (f_b (f_a (f_b (f_c (f_c (f_f (f_f (f_c (f_f (f_f (f_b (f_f (f_a x)))))))))))))))))]
  conv => lhs; rw [h11 (f_a (f_b (f_c (f_c (f_f (f_f (f_c (f_f (f_f (f_b (f_f (f_a x))))))))))))]
  conv => lhs; rw [h11 (f_a (f_e (f_f (f_f (f_e (f_c (f_e (f_d (f_f (f_f (f_a (f_b (f_c (f_c (f_f (f_f (f_c (f_f (f_f (f_b (f_f (f_a x))))))))))))))))))))))]
  conv => lhs; rw [h7 (f_f (f_f (f_e (f_c (f_e (f_d (f_f (f_f (f_a (f_b (f_c (f_c (f_f (f_f (f_c (f_f (f_f (f_b (f_f (f_a x))))))))))))))))))))]
  conv => lhs; rw [h0 (f_f (f_e (f_c (f_e (f_d (f_f (f_f (f_a (f_b (f_c (f_c (f_f (f_f (f_c (f_f (f_f (f_b (f_f (f_a x)))))))))))))))))))]
  conv => lhs; rw [<- h11 (f_c (f_f (f_f (f_b (f_f (f_a x))))))]
  conv => lhs; rw [<- h11 (f_b (f_f (f_a x)))]
  conv => lhs; rw [<- h11 (f_b (f_f (f_f (f_e (f_c (f_e (f_d (f_f (f_f (f_a (f_b (f_c (f_c (f_d (f_b (f_c (f_d (f_b (f_b (f_f (f_a x)))))))))))))))))))))]

theorem c_sym_prov_readout_113_B512 (A : Type) (f_a f_b f_c f_d f_e f_f : A -> A) (x : A) (h0 : forall z : A, f_c (f_f z) = f_b (f_f z)) (h1 : forall z : A, f_b (f_e z) = f_a (f_d z)) (h2 : forall z : A, f_d (f_e (f_e z)) = f_d (f_f (f_a z))) (h3 : forall z : A, f_e (f_f z) = f_e (f_e z)) (h4 : forall z : A, f_d (f_c z) = f_a (f_b z)) (h5 : forall z : A, f_e (f_c (f_c z)) = f_e (f_b (f_d z))) (h6 : forall z : A, f_b (f_b (f_c z)) = f_f (f_f (f_e z))) (h7 : forall z : A, f_a (f_e z) = f_f (f_c z)) (h8 : forall z : A, f_f (f_a z) = f_f (f_e z)) (h9 : forall z : A, f_d (f_e (f_d z)) = f_e (f_f (f_b z))) (h10 : forall z : A, f_a (f_a z) = f_c (f_d z)) (h11 : forall z : A, f_d (f_b z) = f_f (f_f z)) : f_d (f_b (f_f (f_c (f_e (f_f (f_a (f_c (f_e (f_d (f_d (f_b (f_a (f_b (f_c (f_b (f_d (f_b (f_c (f_d (f_e (f_d (f_f (f_e x))))))))))))))))))))))) = f_f (f_d (f_b (f_b (f_f (f_f (f_e (f_c (f_e (f_d (f_f (f_f (f_a (f_b (f_c (f_c (f_d (f_b (f_c (f_d (f_b (f_b (f_f (f_a x))))))))))))))))))))))) := by
  conv => lhs; rw [h11 (f_c (f_d (f_e (f_d (f_f (f_e x))))))]
  conv => lhs; rw [<- h7 (f_e (f_f (f_a (f_c (f_e (f_d (f_d (f_b (f_a (f_b (f_c (f_b (f_f (f_f (f_c (f_d (f_e (f_d (f_f (f_e x))))))))))))))))))))]
  conv => lhs; rw [<- h3 (f_f (f_a (f_c (f_e (f_d (f_d (f_b (f_a (f_b (f_c (f_b (f_f (f_f (f_c (f_d (f_e (f_d (f_f (f_e x)))))))))))))))))))]
  conv => lhs; rw [h11 (f_a (f_e (f_f (f_f (f_a (f_c (f_e (f_d (f_d (f_b (f_a (f_b (f_c (f_b (f_f (f_f (f_c (f_d (f_e (f_d (f_f (f_e x))))))))))))))))))))))]
  conv => lhs; rw [<- h7 (f_d (f_e (f_d (f_f (f_e x)))))]
  conv => lhs; rw [h7 (f_f (f_f (f_a (f_c (f_e (f_d (f_d (f_b (f_a (f_b (f_c (f_b (f_f (f_a (f_e (f_d (f_e (f_d (f_f (f_e x))))))))))))))))))))]
  conv => lhs; rw [<- h11 (f_c (f_f (f_f (f_a (f_c (f_e (f_d (f_d (f_b (f_a (f_b (f_c (f_b (f_f (f_a (f_e (f_d (f_e (f_d (f_f (f_e x)))))))))))))))))))))]
  conv => lhs; rw [h9 (f_f (f_e x))]
  conv => lhs; rw [<- h3 (f_f (f_b (f_f (f_e x))))]
  conv => lhs; rw [h0 (f_f (f_a (f_c (f_e (f_d (f_d (f_b (f_a (f_b (f_c (f_b (f_f (f_a (f_e (f_f (f_f (f_b (f_f (f_e x)))))))))))))))))))]
  conv => lhs; rw [h11 (f_a (f_b (f_c (f_b (f_f (f_a (f_e (f_f (f_f (f_b (f_f (f_e x))))))))))))]
  conv => lhs; rw [h8 (f_c (f_e (f_d (f_f (f_f (f_a (f_b (f_c (f_b (f_f (f_a (f_e (f_f (f_f (f_b (f_f (f_e x)))))))))))))))))]
  conv => lhs; rw [<- h0 (f_a (f_e (f_f (f_f (f_b (f_f (f_e x)))))))]
  conv => lhs; rw [<- h11 (f_b (f_f (f_e x)))]
  conv => lhs; rw [<- h8 x]
  conv => lhs; rw [h7 (f_d (f_b (f_b (f_f (f_a x)))))]
  conv => lhs; rw [<- h11 (f_c (f_d (f_b (f_b (f_f (f_a x))))))]

end LBF19Cert
#print axioms LBF19Cert.c_sym_prov_readout_107_B512
#print axioms LBF19Cert.c_sym_prov_readout_113_B512
