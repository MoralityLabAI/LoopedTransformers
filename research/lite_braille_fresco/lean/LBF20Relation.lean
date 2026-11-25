/-
LBF-20G: synchronous relation composition and balanced walk constraints.
No claim of general completeness of path consistency or code extraction.
-/
namespace LBF20Relation

abbrev Rel := Nat → Nat → Nat → Nat → Prop
abbrev Assignment := Nat → Nat

def Step (R : Rel) : Rel :=
  fun i j a b => R i j a b ∧ ∀ k, ∃ c, R i k a c ∧ R k j c b

def Iter (R : Rel) : Nat → Rel
  | 0 => R
  | t + 1 => Step (Iter R t)

def Sat (R : Rel) (s : Assignment) : Prop :=
  ∀ i j, R i j (s i) (s j)

theorem step_subset {R : Rel} {i j a b : Nat} (h : Step R i j a b) :
    R i j a b := h.1

theorem step_preserves {R : Rel} {s : Assignment} (h : Sat R s) :
    Sat (Step R) s :=
  fun i j => ⟨h i j, fun k => ⟨s k, h i k, h k j⟩⟩

theorem iter_preserves {R : Rel} {s : Assignment} (h : Sat R s) :
    ∀ t, Sat (Iter R t) s
  | 0 => h
  | t + 1 => step_preserves (iter_preserves h t)

inductive Route : Nat → Nat → Nat → Type where
  | edge (i j : Nat) : Route 0 i j
  | keep {t i j : Nat} : Route t i j → Route (t + 1) i j
  | fork {t i k j : Nat} : Route t i k → Route t k j → Route (t + 1) i j

def Holds (R : Rel) : {t i j : Nat} → Route t i j → Nat → Nat → Prop
  | _, _, _, .edge i j, a, b => R i j a b
  | _, _, _, .keep p, a, b => Holds R p a b
  | _, _, _, .fork p q, a, b => ∃ c, Holds R p a c ∧ Holds R q c b

theorem route_sound {R : Rel} :
    ∀ {t i j : Nat} (p : Route t i j) (a b : Nat),
      Iter R t i j a b → Holds R p a b
  | _, _, _, .edge _ _, _, _, h => h
  | _, _, _, .keep p, a, b, h => route_sound p a b h.1
  | _, _, _, @Route.fork t i k j p q, a, b, h =>
      match h.2 k with
      | ⟨c, hl, hr⟩ => ⟨c, route_sound p a c hl, route_sound q c b hr⟩

def capacity : Nat → Nat
  | 0 => 1
  | t + 1 => capacity t + capacity t

def length : {t i j : Nat} → Route t i j → Nat
  | _, _, _, .edge _ _ => 1
  | _, _, _, .keep p => length p
  | _, _, _, .fork p q => length p + length q

theorem length_bound : ∀ {t i j : Nat} (p : Route t i j), length p ≤ capacity t
  | _, _, _, .edge _ _ => Nat.le_refl 1
  | _, _, _, @Route.keep t i j p =>
      Nat.le_trans (length_bound p) (Nat.le_add_right (capacity t) (capacity t))
  | _, _, _, .fork p q => Nat.add_le_add (length_bound p) (length_bound q)

theorem cycle_exclusion {R : Rel} {t i a : Nat} (p : Route t i i)
    (h : ¬ Holds R p a a) : ¬ Iter R t i i a a :=
  fun h' => h (route_sound p a a h')

end LBF20Relation
#print axioms LBF20Relation.step_subset
#print axioms LBF20Relation.step_preserves
#print axioms LBF20Relation.iter_preserves
#print axioms LBF20Relation.route_sound
#print axioms LBF20Relation.length_bound
#print axioms LBF20Relation.cycle_exclusion
