import Lean

/-!
RAM-06 auxiliary, generated exact-tactic curriculum.
These are SYNTHETIC training/dev examples, never original MathCan theorem
coverage. This fixture is read only by the TRAIN job and its Lean checker.
Every candidate script label is independently produced by the actual Lean
4.35 tactic interpreter, then replayed in a fresh Lean process.
-/

namespace RAM06ExactCurriculum

theorem exact_train_prop (P : Prop) (hP : P) : P := by
  exact hP

theorem exact_train_prop_two (P Q : Prop) (hQ : Q) : Q := by
  exact hQ

theorem exact_train_nat_equality (x y : Nat) (hxy : x = y) : x = y := by
  exact hxy

theorem exact_train_int_symm (a b : Int) (hab : a = b) : b = a := by
  exact hab.symm

theorem exact_train_nat_symm (m n : Nat) (hmn : m = n) : n = m := by
  exact hmn.symm

theorem exact_train_conj_left (P Q : Prop) (pair : P ∧ Q) : P := by
  exact pair.left

theorem exact_train_conj_right (P Q : Prop) (both : P ∧ Q) : Q := by
  exact both.right

theorem exact_train_implication (P Q : Prop) (hpq : P → Q) (hp : P) : Q := by
  exact hpq hp

theorem exact_train_implication_reverse (P Q : Prop) (hp : P) (hfn : P → Q) : Q := by
  exact hfn hp

theorem exact_train_two_arguments (P Q R : Prop)
    (hfun : P → Q → R) (hp : P) (hq : Q) : R := by
  exact hfun hp hq

theorem exact_train_iff_forward (P Q : Prop) (hiff : P ↔ Q) (hp : P) : Q := by
  exact hiff.mp hp

theorem exact_train_iff_backward (P Q : Prop) (hiff : P ↔ Q) (hq : Q) : P := by
  exact hiff.mpr hq

theorem exact_dev_inequality (u v : Int) (bound : u ≤ v) : u ≤ v := by
  exact bound

theorem exact_dev_nat_reverse (i j : Nat) (equality : i = j) : j = i := by
  exact equality.symm

theorem exact_dev_conjunction (L R : Prop) (compound : L ∧ R) : L := by
  exact compound.left

theorem exact_dev_forward (A B : Prop) (proofA : A)
    (mapAB : A → B) : B := by
  exact mapAB proofA

theorem exact_dev_iff_reverse (C D : Prop) (bridge : C ↔ D)
    (proofD : D) : C := by
  exact bridge.mpr proofD

theorem exact_dev_two_arguments (H J K : Prop)
    (proofJ : J) (impl : H → J → K) (proofH : H) : K := by
  exact impl proofH proofJ

end RAM06ExactCurriculum
