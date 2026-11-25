namespace LBF19Cert

theorem c_sym_prov_readout_107_B512 (A : Type) (f_a f_b f_c f_d f_e f_f : A -> A) (x : A) (h0 : forall z : A, f_e (f_f z) = f_b (f_c z)) (h1 : forall z : A, f_f (f_b z) = f_d (f_c z)) (h2 : forall z : A, f_e (f_b z) = f_c (f_c z)) (h3 : forall z : A, f_b (f_e z) = f_b (f_b z)) (h4 : forall z : A, f_a (f_c (f_c z)) = f_c (f_b (f_e z))) (h5 : forall z : A, f_b (f_f (f_c z)) = f_b (f_c (f_e z))) (h6 : forall z : A, f_a (f_c z) = f_e (f_a z)) (h7 : forall z : A, f_b (f_d z) = f_a (f_d z)) (h8 : forall z : A, f_f (f_c (f_b z)) = f_c (f_f (f_f z))) (h9 : forall z : A, f_d (f_a z) = f_c (f_e z)) (h10 : forall z : A, f_f (f_a (f_e z)) = f_e (f_d (f_f z))) (h11 : forall z : A, f_b (f_f z) = f_f (f_f z)) : f_e (f_c (f_d (f_b (f_b (f_b (f_d (f_a (f_a (f_f (f_e (f_a (f_b (f_c (f_e (f_a (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x))))))))))))))))))))))) = f_e (f_c (f_d (f_e (f_b (f_b (f_f (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x))))))))))))))))))))))) := by
  conv => lhs; rw [<- h3 (f_e (f_e (f_f x)))]
  conv => lhs; rw [h9 (f_a (f_f (f_e (f_a (f_b (f_c (f_e (f_a (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x))))))))))))))))]
  conv => lhs; rw [<- h3 (f_b (f_c (f_e (f_a (f_f (f_e (f_a (f_b (f_c (f_e (f_a (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x)))))))))))))))))))]
  conv => lhs; rw [h2 (f_c (f_e (f_a (f_f (f_e (f_a (f_b (f_c (f_e (f_a (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x))))))))))))))))))]
  conv => lhs; rw [<- h2 (f_e (f_a (f_f (f_e (f_a (f_b (f_c (f_e (f_a (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x)))))))))))))))))]
  conv => lhs; rw [h3 (f_a (f_f (f_e (f_a (f_b (f_c (f_e (f_a (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x))))))))))))))))]
  conv => lhs; rw [<- h5 (f_b (f_b (f_a (f_f (f_e (f_a (f_b (f_c (f_e (f_a (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x))))))))))))))))))]
  conv => lhs; rw [<- h7 (f_f (f_a (f_b (f_e (f_e (f_e (f_f x)))))))]
  conv => lhs; rw [<- h6 (f_b (f_c (f_e (f_b (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x))))))))))))]
  conv => lhs; rw [h2 (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x))))))))]
  conv => lhs; rw [h8 (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x)))))))))))))))))]
  conv => lhs; rw [<- h0 (f_f (f_f (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x)))))))))))))))))))]
  conv => lhs; rw [<- h11 (f_f (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x))))))))))))))))))]
  conv => lhs; rw [<- h11 (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x)))))))))))))))))]

theorem c_sym_prov_readout_109_B512 (A : Type) (f_a f_b f_c f_d f_e f_f : A -> A) (x : A) (h0 : forall z : A, f_e (f_f z) = f_b (f_c z)) (h1 : forall z : A, f_f (f_b z) = f_d (f_c z)) (h2 : forall z : A, f_e (f_b z) = f_c (f_c z)) (h3 : forall z : A, f_b (f_e z) = f_b (f_b z)) (h4 : forall z : A, f_a (f_c (f_c z)) = f_c (f_b (f_e z))) (h5 : forall z : A, f_b (f_f (f_c z)) = f_b (f_c (f_e z))) (h6 : forall z : A, f_a (f_c z) = f_e (f_a z)) (h7 : forall z : A, f_b (f_d z) = f_a (f_d z)) (h8 : forall z : A, f_f (f_c (f_b z)) = f_c (f_f (f_f z))) (h9 : forall z : A, f_d (f_a z) = f_c (f_e z)) (h10 : forall z : A, f_f (f_a (f_e z)) = f_e (f_d (f_f z))) (h11 : forall z : A, f_b (f_f z) = f_f (f_f z)) : f_e (f_c (f_d (f_b (f_b (f_b (f_d (f_a (f_a (f_f (f_e (f_a (f_b (f_c (f_e (f_a (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x))))))))))))))))))))))) = f_e (f_c (f_d (f_e (f_b (f_b (f_f (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x))))))))))))))))))))))) := by
  conv => lhs; rw [<- h3 (f_b (f_d (f_a (f_a (f_f (f_e (f_a (f_b (f_c (f_e (f_a (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x)))))))))))))))))))]
  conv => lhs; rw [h2 (f_d (f_a (f_a (f_f (f_e (f_a (f_b (f_c (f_e (f_a (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x))))))))))))))))))]
  conv => lhs; rw [<- h3 (f_e (f_e (f_f x)))]
  conv => lhs; rw [h9 (f_a (f_f (f_e (f_a (f_b (f_c (f_e (f_a (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x))))))))))))))))]
  conv => lhs; rw [<- h2 (f_e (f_a (f_f (f_e (f_a (f_b (f_c (f_e (f_a (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x)))))))))))))))))]
  conv => lhs; rw [<- h7 (f_f (f_a (f_b (f_e (f_e (f_e (f_f x)))))))]
  conv => lhs; rw [h3 (f_a (f_f (f_e (f_a (f_b (f_c (f_e (f_b (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x))))))))))))))))]
  conv => lhs; rw [<- h6 (f_b (f_c (f_e (f_b (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x))))))))))))]
  conv => lhs; rw [h2 (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x))))))))]
  conv => lhs; rw [<- h5 (f_b (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x))))))))))))))))))]
  conv => lhs; rw [h8 (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x)))))))))))))))))]
  conv => lhs; rw [<- h0 (f_f (f_f (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x)))))))))))))))))))]
  conv => lhs; rw [<- h11 (f_f (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x))))))))))))))))))]
  conv => lhs; rw [<- h11 (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x)))))))))))))))))]

theorem c_sym_prov_readout_113_B512 (A : Type) (f_a f_b f_c f_d f_e f_f : A -> A) (x : A) (h0 : forall z : A, f_e (f_f z) = f_b (f_c z)) (h1 : forall z : A, f_f (f_b z) = f_d (f_c z)) (h2 : forall z : A, f_e (f_b z) = f_c (f_c z)) (h3 : forall z : A, f_b (f_e z) = f_b (f_b z)) (h4 : forall z : A, f_a (f_c (f_c z)) = f_c (f_b (f_e z))) (h5 : forall z : A, f_b (f_f (f_c z)) = f_b (f_c (f_e z))) (h6 : forall z : A, f_a (f_c z) = f_e (f_a z)) (h7 : forall z : A, f_b (f_d z) = f_a (f_d z)) (h8 : forall z : A, f_f (f_c (f_b z)) = f_c (f_f (f_f z))) (h9 : forall z : A, f_d (f_a z) = f_c (f_e z)) (h10 : forall z : A, f_f (f_a (f_e z)) = f_e (f_d (f_f z))) (h11 : forall z : A, f_b (f_f z) = f_f (f_f z)) : f_e (f_c (f_d (f_b (f_b (f_b (f_d (f_a (f_a (f_f (f_e (f_a (f_b (f_c (f_e (f_a (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x))))))))))))))))))))))) = f_e (f_c (f_d (f_e (f_b (f_b (f_f (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_e (f_e (f_e (f_f x))))))))))))))))))))))) := by
  conv => lhs; rw [<- h7 (f_f (f_a (f_b (f_b (f_e (f_e (f_f x)))))))]
  conv => lhs; rw [h2 (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x))))))))]
  conv => lhs; rw [<- h6 (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x))))))))))))]
  conv => lhs; rw [<- h3 (f_b (f_d (f_a (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x)))))))))))))))))))]
  conv => lhs; rw [h9 (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x))))))))))))))))]
  conv => lhs; rw [h2 (f_c (f_e (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x))))))))))))))))))]
  conv => lhs; rw [<- h2 (f_e (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x)))))))))))))))))]
  conv => lhs; rw [h3 (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x))))))))))))))))]
  conv => lhs; rw [<- h5 (f_b (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x))))))))))))))))))]
  conv => lhs; rw [h8 (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x)))))))))))))))))]
  conv => lhs; rw [<- h0 (f_f (f_f (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x)))))))))))))))))))]
  conv => lhs; rw [<- h11 (f_f (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x))))))))))))))))))]
  conv => lhs; rw [<- h11 (f_b (f_a (f_f (f_a (f_c (f_b (f_c (f_c (f_c (f_d (f_f (f_a (f_b (f_b (f_e (f_e (f_f x)))))))))))))))))]
  conv => lhs; rw [<- h3 (f_e (f_e (f_f x)))]

end LBF19Cert
#print axioms LBF19Cert.c_sym_prov_readout_107_B512
#print axioms LBF19Cert.c_sym_prov_readout_109_B512
#print axioms LBF19Cert.c_sym_prov_readout_113_B512
