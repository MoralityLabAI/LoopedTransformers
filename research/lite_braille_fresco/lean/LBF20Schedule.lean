/-
LBF-20H: sound asynchronous revisions and schedule-independent fixed points.
The finite dependency queue and bitset implementation are audited separately.
-/
namespace LBF20Schedule

abbrev Rel := Nat → Nat → Nat → Nat → Prop
abbrev Assignment := Nat → Nat
abbrev Action := Nat × Nat × Nat

def Sub (R S : Rel) : Prop :=
  ∀ i j a b, R i j a b → S i j a b

def Sat (R : Rel) (s : Assignment) : Prop :=
  ∀ i j, R i j (s i) (s j)

def Fixed (R : Rel) : Prop :=
  ∀ i k j a b, R i j a b → ∃ c, R i k a c ∧ R k j c b

def Revise (R : Rel) (i k j : Nat) : Rel :=
  fun x y a b => R x y a b ∧
    (x = i → y = j → ∃ c, R i k a c ∧ R k j c b)

theorem revise_subset (R : Rel) (i k j : Nat) :
    Sub (Revise R i k j) R := fun _ _ _ _ h => h.1

theorem revise_mono {R S : Rel} (h : Sub R S) (i k j : Nat) :
    Sub (Revise R i k j) (Revise S i k j) := by
  intro x y a b p
  refine ⟨h _ _ _ _ p.1, ?_⟩
  intro hx hy
  obtain ⟨c, hc, hd⟩ := p.2 hx hy
  exact ⟨c, h _ _ _ _ hc, h _ _ _ _ hd⟩

theorem revise_preserves {R : Rel} {s : Assignment} (h : Sat R s)
    (i k j : Nat) : Sat (Revise R i k j) s := by
  intro x y
  refine ⟨h x y, ?_⟩
  intro hx hy
  subst x
  subst y
  exact ⟨s k, h i k, h k j⟩

theorem skip_safe {R : Rel} (i k j : Nat)
    (h : ∀ a b, R i j a b → ∃ c, R i k a c ∧ R k j c b) :
    Sub R (Revise R i k j) := by
  intro x y a b p
  refine ⟨p, ?_⟩
  intro hx hy
  subst x
  subst y
  exact h _ _ p

theorem fixed_survives {F R : Rel} (hf : Fixed F) (hs : Sub F R)
    (i k j : Nat) : Sub F (Revise R i k j) := by
  intro x y a b p
  refine ⟨hs _ _ _ _ p, ?_⟩
  intro hx hy
  subst x
  subst y
  obtain ⟨c, hc, hd⟩ := hf i k j a b p
  exact ⟨c, hs _ _ _ _ hc, hs _ _ _ _ hd⟩

def Run (R : Rel) : List Action → Rel
  | [] => R
  | (i, k, j) :: rest => Run (Revise R i k j) rest

theorem run_preserves {R : Rel} {s : Assignment} (h : Sat R s)
    (schedule : List Action) : Sat (Run R schedule) s := by
  induction schedule generalizing R with
  | nil => exact h
  | cons a rest ih =>
      rcases a with ⟨i,k,j⟩
      exact ih (revise_preserves h i k j)

theorem run_below (R : Rel) (schedule : List Action) : Sub (Run R schedule) R := by
  induction schedule generalizing R with
  | nil => exact fun _ _ _ _ h => h
  | cons a rest ih =>
      rcases a with ⟨i,k,j⟩
      intro x y u v h
      exact (ih (Revise R i k j) x y u v h).1

theorem run_contains_fixed {F R : Rel} (hf : Fixed F) (hs : Sub F R)
    (schedule : List Action) : Sub F (Run R schedule) := by
  induction schedule generalizing R with
  | nil => exact hs
  | cons a rest ih =>
      rcases a with ⟨i,k,j⟩
      exact ih (fixed_survives hf hs i k j)

theorem fixed_terminal_unique {R : Rel} (a b : List Action)
    (ha : Fixed (Run R a)) (hb : Fixed (Run R b)) :
    Sub (Run R a) (Run R b) ∧ Sub (Run R b) (Run R a) :=
  ⟨run_contains_fixed ha (run_below R a) b,
    run_contains_fixed hb (run_below R b) a⟩

end LBF20Schedule
#print axioms LBF20Schedule.revise_subset
#print axioms LBF20Schedule.revise_mono
#print axioms LBF20Schedule.revise_preserves
#print axioms LBF20Schedule.skip_safe
#print axioms LBF20Schedule.fixed_survives
#print axioms LBF20Schedule.run_preserves
#print axioms LBF20Schedule.run_below
#print axioms LBF20Schedule.run_contains_fixed
#print axioms LBF20Schedule.fixed_terminal_unique
