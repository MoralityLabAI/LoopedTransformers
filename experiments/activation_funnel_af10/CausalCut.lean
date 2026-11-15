/-!
AF-10: kernel-checked finite-coordinate abstraction of a downstream causal cut.
This proves a structural projection identity for arbitrary update functions.
The Python/Torch tests separately establish that the concrete role-indexed
code restoration implements this map and preserves the real status readout.
No neural theorem-proving claim follows from this abstract Lean proof.
-/

def cutStep {n : Nat} {α : Type} (keep : Fin n → Bool)
    (source : Fin n → α) (update : (Fin n → α) → Fin n → α)
    (state : Fin n → α) : Fin n → α :=
  fun i => if keep i then source i else update state i

def evolve {n : Nat} {α : Type} (keep : Fin n → Bool)
    (source : Fin n → α) (update : (Fin n → α) → Fin n → α) :
    Nat → Fin n → α
  | 0 => source
  | t + 1 => cutStep keep source update (evolve keep source update t)

theorem cut_keeps_source {n : Nat} {α : Type}
    (keep : Fin n → Bool) (source : Fin n → α)
    (update : (Fin n → α) → Fin n → α) (i : Fin n)
    (h : keep i = true) (t : Nat) :
    evolve keep source update t i = source i := by
  cases t with
  | zero => rfl
  | succ t => simp [evolve, cutStep, h]

theorem cut_noninterference {n : Nat} {α : Type}
    (keep : Fin n → Bool) (recipient donor : Fin n → α)
    (update : (Fin n → α) → Fin n → α) (i : Fin n)
    (h : keep i = true) (same : recipient i = donor i) (t : Nat) :
    evolve keep recipient update t i = evolve keep donor update t i := by
  rw [cut_keeps_source keep recipient update i h t]
  rw [cut_keeps_source keep donor update i h t]
  exact same

def transplant {n : Nat} {α : Type} (j : Fin n)
    (recipient donor : Fin n → α) : Fin n → α :=
  fun i => if i == j then donor i else recipient i

theorem transplant_other_unchanged {n : Nat} {α : Type}
    (recipient donor : Fin n → α) (j i : Fin n)
    (different : i ≠ j) :
    transplant j recipient donor i = recipient i := by
  simp [transplant, different]

theorem cut_transplant_protected {n : Nat} {α : Type}
    (keep : Fin n → Bool) (recipient donor : Fin n → α)
    (update : (Fin n → α) → Fin n → α) (j i : Fin n)
    (h : keep i = true) (different : i ≠ j) (t : Nat) :
    evolve keep (transplant j recipient donor) update t i =
      evolve keep recipient update t i := by
  apply cut_noninterference keep (transplant j recipient donor) recipient update i h
  exact transplant_other_unchanged recipient donor j i different
