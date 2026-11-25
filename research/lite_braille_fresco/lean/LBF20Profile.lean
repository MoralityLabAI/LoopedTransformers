/-
LBF-20L: profile congruence and label-preserving prefix quotienting.
Local relational identities, not executable refinement.
-/
namespace LBF20Profile

abbrev Domains := Nat → Nat → Prop
abbrev Relations := Nat → Nat → Nat → Nat → Prop
abbrev Assignment := Nat → Nat
abbrev Scope := Nat → Bool

def Restrict (D : Domains) (R : Relations) (j a : Nat) : Domains :=
  fun i b => D i b ∧ R j i a b

def Residual (D : Domains) (R : Relations) (C : Scope)
    (x : Assignment) : Domains :=
  fun i b => D i b ∧ ∀ j, C j = true → R j i (x j) b

theorem profile_congruence {D : Domains} {R : Relations} {C : Scope}
    {x y : Assignment}
    (h : ∀ j i b, C j = true → D i b →
      (R j i (x j) b ↔ R j i (y j) b)) (i b : Nat) :
    Residual D R C x i b ↔ Residual D R C y i b := by
  constructor
  · intro ⟨hd, hr⟩
    exact ⟨hd, fun j hj => (h j i b hj hd).mp (hr j hj)⟩
  · intro ⟨hd, hr⟩
    exact ⟨hd, fun j hj => (h j i b hj hd).mpr (hr j hj)⟩

theorem restrict_commute (D : Domains) (R : Relations) (j a k c i b : Nat) :
    Restrict (Restrict D R j a) R k c i b ↔
    Restrict (Restrict D R k c) R j a i b :=
  ⟨fun h => ⟨⟨h.1.1, h.2⟩, h.1.2⟩,
    fun h => ⟨⟨h.1.1, h.2⟩, h.1.2⟩⟩

theorem restrict_idempotent (D : Domains) (R : Relations) (j a i b : Nat) :
    Restrict (Restrict D R j a) R j a i b ↔ Restrict D R j a i b :=
  ⟨fun h => h.1, fun h => ⟨h,h.2⟩⟩

theorem restrict_congruence {D E : Domains} (R : Relations)
    (h : ∀ i b, D i b ↔ E i b) (j a i b : Nat) :
    Restrict D R j a i b ↔ Restrict E R j a i b :=
  ⟨fun h' => ⟨(h i b).mp h'.1,h'.2⟩,
    fun h' => ⟨(h i b).mpr h'.1,h'.2⟩⟩

theorem label_transition {L K V : Type} (key : L → K)
    (extend : L → V → L) (step : K → V → K)
    (h : ∀ a v, key (extend a v) = step (key a) v)
    {a b : L} (e : key a = key b) (v : V) :
    key (extend a v) = key (extend b v) :=
  (h a v).trans ((congrArg (fun k => step k v) e).trans (h b v).symm)

theorem labeled_quotient {L K X : Type} (key : L → K)
    (R : L → X → Prop) (G : K → X → Prop)
    (h : ∀ a x, R a x ↔ G (key a) x)
    (evidence : L → Prop) (query : X → Prop) (x : X) :
    (∃ a, evidence a ∧ R a x ∧ query x) ↔
    (∃ k, (∃ a, evidence a ∧ key a = k) ∧ G k x ∧ query x) := by
  constructor
  · intro ⟨a, he, hr, hq⟩
    exact ⟨key a, ⟨a,he,rfl⟩, (h a x).mp hr,hq⟩
  · intro ⟨k, ⟨a,he,hk⟩, hg,hq⟩
    exact ⟨a,he,(h a x).mpr (hk.symm ▸ hg),hq⟩

end LBF20Profile
#print axioms LBF20Profile.profile_congruence
#print axioms LBF20Profile.restrict_commute
#print axioms LBF20Profile.restrict_idempotent
#print axioms LBF20Profile.restrict_congruence
#print axioms LBF20Profile.label_transition
#print axioms LBF20Profile.labeled_quotient
