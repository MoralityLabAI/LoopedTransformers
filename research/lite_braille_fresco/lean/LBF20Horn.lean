/-
LBF-20E: context-scoped Horn proof supports.
The table characterization is conditional on semantic soundness and saturation.
No executable extraction claim.
-/
namespace LBF20Horn

structure Clause where
  head : Nat
  body : List Nat

abbrev Rules := Nat → Clause
abbrev Mask := Nat → Prop

def Sub (a b : Mask) : Prop := ∀ x, a x → b x

inductive Derives (R : Rules) (F M : Mask) : Nat → Prop where
  | fact {a : Nat} : F a → Derives R F M a
  | fire (i : Nat) : M i →
      (∀ a, a ∈ (R i).body → Derives R F M a) → Derives R F M (R i).head

theorem monotone {R : Rules} {F G M N : Mask}
    (hf : Sub F G) (hm : Sub M N) {a : Nat}
    (d : Derives R F M a) : Derives R G N a := by
  induction d with
  | fact h => exact .fact (hf _ h)
  | fire i h children ih => exact .fire i (hm _ h) ih

/-- A cache fact may be inlined if it is derivable in the receiving context. -/
theorem inline_cache {R : Rules} {F G M : Mask}
    (h : ∀ a, G a → Derives R F M a) {a : Nat}
    (d : Derives R G M a) : Derives R F M a := by
  induction d with
  | fact ha => exact h _ ha
  | fire i hi children ih => exact .fire i hi ih

theorem cache_extension {R : Rules} {F C M : Mask}
    (hc : ∀ a, C a → Derives R F M a) (a : Nat) :
    Derives R (fun x => F x ∨ C x) M a ↔ Derives R F M a := by
  constructor
  · intro d
    exact inline_cache (fun x h => h.elim (fun f => Derives.fact f) (hc x)) d
  · intro d
    exact monotone (fun _ h => Or.inl h) (fun _ h => h) d

theorem cache_context_inclusion {R : Rules} {F G M : Mask}
    (h : Sub F G) {a : Nat} (d : Derives R F M a) : Derives R G M a :=
  monotone h (fun _ p => p) d

abbrev Table := Nat → Mask → Prop

def Covered (T : Table) (M : Mask) (a : Nat) : Prop :=
  ∃ N, T a N ∧ Sub N M

structure Saturated (R : Rules) (F : Mask) (T : Table) : Prop where
  fact : ∀ a, F a → T a (fun _ => False)
  fire : ∀ (i : Nat) (M : Mask), M i →
    (∀ a, a ∈ (R i).body → Covered T M a) → Covered T M (R i).head

theorem derivation_covered {R : Rules} {F M : Mask} {T : Table}
    (h : Saturated R F T) {a : Nat} (d : Derives R F M a) : Covered T M a := by
  induction d with
  | fact hf => exact ⟨fun _ => False, h.fact _ hf, fun _ z => False.elim z⟩
  | fire i hi children ih => exact h.fire i M hi ih

theorem support_query_exact {R : Rules} {F M : Mask} {T : Table}
    (hs : ∀ a N, T a N → Derives R F N a)
    (hc : Saturated R F T) (a : Nat) :
    Covered T M a ↔ Derives R F M a := by
  constructor
  · intro ⟨N, hn, sub⟩
    exact monotone (fun _ h => h) sub (hs a N hn)
  · exact derivation_covered hc

theorem no_empty_context_proof {R : Rules} {a : Nat}
    (d : Derives R (fun _ => False) (fun _ => False) a) : False := by
  cases d with
  | fact h => exact h
  | fire i h children => exact h

theorem context_cache_counterexample (R : Rules) :
    Derives R (fun x => x = 0) (fun _ => False) 0 ∧
    ¬ Derives R (fun _ => False) (fun _ => False) 0 :=
  ⟨Derives.fact rfl, no_empty_context_proof⟩

end LBF20Horn
#print axioms LBF20Horn.monotone
#print axioms LBF20Horn.inline_cache
#print axioms LBF20Horn.cache_extension
#print axioms LBF20Horn.cache_context_inclusion
#print axioms LBF20Horn.derivation_covered
#print axioms LBF20Horn.support_query_exact
#print axioms LBF20Horn.no_empty_context_proof
#print axioms LBF20Horn.context_cache_counterexample
