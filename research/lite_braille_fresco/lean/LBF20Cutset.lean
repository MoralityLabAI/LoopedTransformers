/-
LBF-20K: exact support unions over cutset assignments.
Local relational semantics; the finite conditioned forest is independently audited.
-/
namespace LBF20Cutset
abbrev Assignment := Nat → Nat
abbrev Scope := Nat → Bool
abbrev Table := Assignment → Prop

def Agree (C : Scope) (x y : Assignment) : Prop :=
  ∀ i, C i = true → x i = y i

def Project (C : Scope) (P : Table) : Table :=
  fun x => ∃ y, P y ∧ Agree C x y

def Branch (C : Scope) (a : Assignment) (P : Table) : Table :=
  fun x => P x ∧ Agree C x a

theorem branch_cover (C : Scope) (P : Table) (x : Assignment) :
    P x ↔ ∃ a, Branch C a P x :=
  ⟨fun h => ⟨x, h, fun _ _ => rfl⟩, fun ⟨_, h, _⟩ => h⟩

theorem incompatible_branches {C : Scope} {a b x : Assignment} {P : Table}
    (h : Branch C a P x) (g : Branch C b P x) :
    Agree C a b :=
  fun i hi => (h.2 i hi).symm.trans (g.2 i hi)

theorem support_union (C S : Scope) (P : Table) (x : Assignment) :
    Project S P x ↔ ∃ a, Project S (Branch C a P) x := by
  constructor
  · intro ⟨y, h, e⟩
    exact ⟨y, y, ⟨h, fun _ _ => rfl⟩, e⟩
  · intro ⟨a, y, h, e⟩
    exact ⟨y, h.1, e⟩

def Covered (C : Scope) (A : Assignment → Prop) (P : Table) : Prop :=
  ∀ x, P x → ∃ a, A a ∧ Agree C x a

theorem restricted_union {C S : Scope} {A : Assignment → Prop} {P : Table}
    (h : Covered C A P) (x : Assignment) :
    Project S P x ↔ ∃ a, A a ∧ Project S (Branch C a P) x := by
  constructor
  · intro ⟨y, py, e⟩
    obtain ⟨a, ha, e'⟩ := h y py
    exact ⟨a, ha, y, ⟨py, e'⟩, e⟩
  · intro ⟨a, ha, y, hy, e⟩
    exact ⟨y, hy.1, e⟩

/-- Quotient equal residual tables, retaining the compatible branch labels. -/
theorem labeled_group_exact {Label : Type} {R : Label → Table}
    {G : Table} {A : Label → Prop} {E : Table}
    (h : ∀ a, A a → ∀ x, R a x ↔ G x) (x : Assignment) :
    (∃ a, A a ∧ R a x ∧ E x) ↔ (∃ a, A a) ∧ G x ∧ E x := by
  constructor
  · intro ⟨a, ha, hr, he⟩
    exact ⟨⟨a, ha⟩, (h a ha x).mp hr, he⟩
  · intro ⟨⟨a, ha⟩, hg, he⟩
    exact ⟨a, ha, (h a ha x).mpr hg, he⟩

theorem conditional_union (C S : Scope) (P E : Table) (x : Assignment) :
    Project S (fun y => P y ∧ E y) x ↔
    ∃ a, Project S (Branch C a (fun y => P y ∧ E y)) x :=
  support_union C S (fun y => P y ∧ E y) x

end LBF20Cutset
#print axioms LBF20Cutset.branch_cover
#print axioms LBF20Cutset.incompatible_branches
#print axioms LBF20Cutset.support_union
#print axioms LBF20Cutset.restricted_union
#print axioms LBF20Cutset.labeled_group_exact
#print axioms LBF20Cutset.conditional_union
