namespace LBF19Cert

theorem c_sym_prov_readout_107_B512 (A : Type) (f_a f_b f_c f_d f_e f_f : A -> A) (x : A) (h0 : forall z : A, f_d (f_e z) = f_a (f_d z)) (h1 : forall z : A, f_d (f_d z) = f_f (f_a z)) (h2 : forall z : A, f_d (f_c z) = f_c (f_d z)) (h3 : forall z : A, f_a (f_d (f_c z)) = f_c (f_a (f_f z))) (h4 : forall z : A, f_c (f_f z) = f_c (f_e z)) (h5 : forall z : A, f_c (f_c (f_d z)) = f_a (f_c (f_d z))) (h6 : forall z : A, f_d (f_b z) = f_c (f_a z)) (h7 : forall z : A, f_a (f_f z) = f_c (f_c z)) (h8 : forall z : A, f_e (f_d z) = f_f (f_b z)) (h9 : forall z : A, f_b (f_a z) = f_b (f_d z)) (h10 : forall z : A, f_d (f_b (f_d z)) = f_c (f_d (f_a z))) (h11 : forall z : A, f_b (f_e (f_f z)) = f_a (f_e (f_c z))) : f_c (f_c (f_d (f_e (f_c (f_e (f_e (f_e (f_d (f_d (f_c (f_f (f_d (f_c (f_e (f_b x))))))))))))))) = f_a (f_c (f_d (f_f (f_c (f_e (f_e (f_e (f_c (f_a (f_c (f_a (f_d (f_c (f_f (f_b x))))))))))))))) := by
  conv => lhs; rw [h5 (f_e (f_c (f_e (f_e (f_e (f_d (f_d (f_c (f_f (f_d (f_c (f_e (f_b x)))))))))))))]
  conv => lhs; rw [h4 (f_d (f_c (f_e (f_b x))))]
  conv => lhs; rw [h2 (f_e (f_d (f_c (f_e (f_b x)))))]
  conv => lhs; rw [h2 (f_d (f_e (f_d (f_c (f_e (f_b x))))))]
  conv => lhs; rw [h0 (f_d (f_c (f_e (f_b x))))]
  conv => lhs; rw [<- h10 (f_d (f_d (f_c (f_e (f_b x)))))]
  conv => lhs; rw [<- h4 (f_b x)]
  conv => lhs; rw [h6 (f_d (f_d (f_d (f_c (f_f (f_b x))))))]
  conv => lhs; rw [h1 (f_d (f_c (f_f (f_b x))))]
  conv => lhs; rw [h7 (f_a (f_d (f_c (f_f (f_b x)))))]
  conv => lhs; rw [<- h0 (f_c (f_f (f_b x)))]
  conv => lhs; rw [h5 (f_e (f_c (f_f (f_b x))))]
  conv => lhs; rw [h0 (f_c (f_f (f_b x)))]
  conv => lhs; rw [<- h2 (f_e (f_c (f_e (f_e (f_e (f_c (f_a (f_c (f_a (f_d (f_c (f_f (f_b x)))))))))))))]
  conv => lhs; rw [<- h4 (f_c (f_e (f_e (f_e (f_c (f_a (f_c (f_a (f_d (f_c (f_f (f_b x))))))))))))]
  conv => lhs; rw [h2 (f_f (f_c (f_e (f_e (f_e (f_c (f_a (f_c (f_a (f_d (f_c (f_f (f_b x)))))))))))))]

end LBF19Cert
#print axioms LBF19Cert.c_sym_prov_readout_107_B512
