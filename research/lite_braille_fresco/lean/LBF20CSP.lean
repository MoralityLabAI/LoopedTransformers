/-
LBF-20F: arc and pair propagation preserve every satisfying assignment.
Relational semantics, independent of the finite bitset implementation.
-/
namespace LBF20CSP
abbrev Domains := Nat → Nat → Prop
abbrev Relations := Nat → Nat → Nat → Nat → Prop
abbrev Assignment := Nat → Nat

def Allowed (D : Domains) (s : Assignment) : Prop := ∀ i, D i (s i)
def Satisfies (R : Relations) (s : Assignment) : Prop := ∀ i j, R i j (s i) (s j)

def Arc (D : Domains) (R : Relations) (i j : Nat) : Domains :=
  fun v a => D v a ∧ (v = i → ∃ b, D j b ∧ R i j a b)

def Path (R : Relations) (i j k : Nat) : Relations :=
  fun x y a b => R x y a b ∧ (x = i → y = j → ∃ c, R i k a c ∧ R k j c b)

theorem arc_subset (D : Domains) (R : Relations) (i j v a : Nat) :
    Arc D R i j v a → D v a := fun h => h.1

theorem arc_preserves {D : Domains} {R : Relations} {s : Assignment}
    (hd : Allowed D s) (hr : Satisfies R s) (i j : Nat) : Allowed (Arc D R i j) s := by
  intro v
  refine ⟨hd v, ?_⟩
  intro h
  subst v
  exact ⟨s j, hd j, hr i j⟩

theorem path_subset (R : Relations) (i j k x y a b : Nat) :
    Path R i j k x y a b → R x y a b := fun h => h.1

theorem path_preserves {R : Relations} {s : Assignment}
    (hr : Satisfies R s) (i j k : Nat) : Satisfies (Path R i j k) s := by
  intro x y
  refine ⟨hr x y, ?_⟩
  intro hx hy
  subst x
  subst y
  exact ⟨s k, hr i k, hr k j⟩

theorem independent_arc_support (a : Bool) : ∃ b : Bool, a ≠ b := by
  cases a
  · exact ⟨true, by decide⟩
  · exact ⟨false, by decide⟩

theorem odd_cycle_impossible (a b c : Bool) :
    a ≠ b → b ≠ c → c ≠ a → False := by
  cases a <;> cases b <;> cases c <;> intro hab hbc hca <;>
    first | exact hab rfl | exact hbc rfl | exact hca rfl

end LBF20CSP
#print axioms LBF20CSP.arc_subset
#print axioms LBF20CSP.arc_preserves
#print axioms LBF20CSP.path_subset
#print axioms LBF20CSP.path_preserves
#print axioms LBF20CSP.independent_arc_support
#print axioms LBF20CSP.odd_cycle_impossible
