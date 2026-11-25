namespace LBF19Cert

theorem c_sym_prov_readout_109_B512 (A : Type) (f_a f_b f_c f_d f_e f_f : A -> A) (x : A) (h0 : forall z : A, f_b (f_d z) = f_e (f_f z)) (h1 : forall z : A, f_f (f_a (f_c z)) = f_d (f_a (f_a z))) (h2 : forall z : A, f_b (f_b (f_a z)) = f_b (f_f (f_f z))) (h3 : forall z : A, f_b (f_a z) = f_d (f_c z)) (h4 : forall z : A, f_c (f_c (f_c z)) = f_f (f_d (f_d z))) (h5 : forall z : A, f_d (f_b z) = f_f (f_d z)) (h6 : forall z : A, f_c (f_f z) = f_b (f_e z)) (h7 : forall z : A, f_e (f_f (f_b z)) = f_c (f_f (f_c z))) (h8 : forall z : A, f_a (f_f z) = f_e (f_c z)) (h9 : forall z : A, f_f (f_a z) = f_f (f_f z)) (h10 : forall z : A, f_a (f_e z) = f_b (f_b z)) (h11 : forall z : A, f_d (f_f z) = f_b (f_c z)) : f_a (f_e (f_c (f_b (f_c (f_b (f_c (f_f (f_f (f_b (f_a (f_a (f_a (f_a (f_e (f_e (f_f (f_d (f_e (f_b (f_b (f_e (f_a (f_e x))))))))))))))))))))))) = f_b (f_d (f_f (f_d (f_f (f_f (f_e (f_e (f_d (f_d (f_f (f_a (f_f (f_d (f_f (f_a (f_d (f_b (f_e (f_b (f_b (f_e (f_b (f_b x))))))))))))))))))))))) := by
  conv => lhs; rw [h3 (f_a (f_a (f_a (f_e (f_e (f_f (f_d (f_e (f_b (f_b (f_e (f_a (f_e x)))))))))))))]
  conv => lhs; rw [<- h5 (f_c (f_a (f_a (f_a (f_e (f_e (f_f (f_d (f_e (f_b (f_b (f_e (f_a (f_e x))))))))))))))]
  conv => lhs; rw [<- h11 (f_a (f_a (f_a (f_e (f_e (f_f (f_d (f_e (f_b (f_b (f_e (f_a (f_e x)))))))))))))]
  conv => lhs; rw [h6 (f_d (f_d (f_f (f_a (f_a (f_a (f_e (f_e (f_f (f_d (f_e (f_b (f_b (f_e (f_a (f_e x))))))))))))))))]
  conv => lhs; rw [h10 (f_e (f_f (f_d (f_e (f_b (f_b (f_e (f_a (f_e x)))))))))]
  conv => lhs; rw [h10 (f_c (f_b (f_c (f_b (f_b (f_e (f_d (f_d (f_f (f_a (f_a (f_b (f_b (f_e (f_f (f_d (f_e (f_b (f_b (f_e (f_a (f_e x))))))))))))))))))))))]
  conv => lhs; rw [<- h11 (f_b (f_b (f_e (f_d (f_d (f_f (f_a (f_a (f_b (f_b (f_e (f_f (f_d (f_e (f_b (f_b (f_e (f_a (f_e x)))))))))))))))))))]
  conv => lhs; rw [<- h10 (f_e (f_d (f_d (f_f (f_a (f_a (f_b (f_b (f_e (f_f (f_d (f_e (f_b (f_b (f_e (f_a (f_e x)))))))))))))))))]
  conv => lhs; rw [<- h6 (f_f (f_d (f_e (f_b (f_b (f_e (f_a (f_e x))))))))]
  conv => lhs; rw [<- h11 (f_f (f_f (f_d (f_e (f_b (f_b (f_e (f_a (f_e x)))))))))]
  conv => lhs; rw [h10 x]
  conv => lhs; rw [<- h5 (f_e (f_b (f_b (f_e (f_b (f_b x))))))]
  conv => lhs; rw [h9 (f_a (f_d (f_f (f_f (f_d (f_b (f_e (f_b (f_b (f_e (f_b (f_b x))))))))))))]
  conv => lhs; rw [<- h9 (f_d (f_b (f_e (f_b (f_b (f_e (f_b (f_b x))))))))]
  conv => lhs; rw [h9 (f_e (f_e (f_d (f_d (f_f (f_f (f_a (f_d (f_f (f_a (f_d (f_b (f_e (f_b (f_b (f_e (f_b (f_b x))))))))))))))))))]
  conv => lhs; rw [h9 (f_d (f_f (f_a (f_d (f_b (f_e (f_b (f_b (f_e (f_b (f_b x)))))))))))]
  conv => lhs; rw [<- h9 (f_f (f_d (f_f (f_a (f_d (f_b (f_e (f_b (f_b (f_e (f_b (f_b x))))))))))))]
  conv => lhs; rw [<- h11 (f_d (f_f (f_f (f_e (f_e (f_d (f_d (f_f (f_a (f_f (f_d (f_f (f_a (f_d (f_b (f_e (f_b (f_b (f_e (f_b (f_b x)))))))))))))))))))))]

end LBF19Cert
#print axioms LBF19Cert.c_sym_prov_readout_109_B512
