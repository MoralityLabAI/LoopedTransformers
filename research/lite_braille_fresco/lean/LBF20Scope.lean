/-
LBF-20I: scoped local joins, projections and separator gluing.
Relational statements only; the finite bitset engine is tested separately.
-/
namespace LBF20Scope

abbrev Assignment := Nat → Nat
abbrev Table := Assignment → Prop
abbrev Scope := List Nat

def Agree (A : Scope) (x y : Assignment) : Prop :=
  ∀ i, i ∈ A → x i = y i

def Project (A : Scope) (T : Table) : Table :=
  fun x => ∃ y, T y ∧ Agree A x y

def Sub (P Q : Table) : Prop := ∀ x, P x → Q x

def Local (A : Scope) (T : Table) : Prop :=
  ∀ x y, Agree A x y → T x → T y

def Join (P Q : Table) : Table := fun x => P x ∧ Q x

theorem agree_refl (A : Scope) (x : Assignment) : Agree A x x :=
  fun _ _ => rfl

theorem agree_symm {A : Scope} {x y : Assignment} (h : Agree A x y) :
    Agree A y x := fun i hi => (h i hi).symm

theorem project_contains (A : Scope) (T : Table) : Sub T (Project A T) :=
  fun x h => ⟨x, h, agree_refl A x⟩

theorem project_mono {P Q : Table} (h : Sub P Q) (A : Scope) :
    Sub (Project A P) (Project A Q) :=
  fun _ ⟨y, hy, he⟩ => ⟨y, h y hy, he⟩

theorem join_project_safe (A : Scope) (P Q : Table) :
    Sub (Join P Q) (Join P (Project A (Join P Q))) :=
  fun x h => ⟨h.1, project_contains A (Join P Q) x h⟩

theorem project_local (A : Scope) (P : Table) : Local A (Project A P) := by
  intro x y h p
  obtain ⟨z, hz, he⟩ := p
  exact ⟨z, hz, fun i hi => (h i hi).symm.trans (he i hi)⟩

theorem local_projection_exact {A : Scope} {P : Table} (h : Local A P) :
    Sub (Project A P) P :=
  fun x ⟨y, hy, he⟩ => h y x (agree_symm he) hy

def Glue (A : Scope) (x y : Assignment) : Assignment :=
  fun i => if i ∈ A then x i else y i

theorem glue_left (A : Scope) (x y : Assignment) : Agree A x (Glue A x y) := by
  intro i hi
  change x i = (if i ∈ A then x i else y i)
  exact (if_pos hi).symm

theorem glue_right {A B : Scope} {x y : Assignment}
    (h : ∀ i, i ∈ A → i ∈ B → x i = y i) : Agree B y (Glue A x y) := by
  intro i hi
  change y i = (if i ∈ A then x i else y i)
  by_cases ha : i ∈ A
  · exact (h i ha hi).symm.trans (if_pos ha).symm
  · exact (if_neg ha).symm

theorem separator_glue {A B : Scope} {P Q : Table} {x y : Assignment}
    (hp : Local A P) (hq : Local B Q)
    (px : P x) (qy : Q y)
    (h : ∀ i, i ∈ A → i ∈ B → x i = y i) :
    ∃ z, P z ∧ Q z ∧ Agree A x z ∧ Agree B y z :=
  ⟨Glue A x y, hp x _ (glue_left A x y) px,
    hq y _ (glue_right h) qy, glue_left A x y, glue_right h⟩

end LBF20Scope
#print axioms LBF20Scope.agree_refl
#print axioms LBF20Scope.agree_symm
#print axioms LBF20Scope.project_contains
#print axioms LBF20Scope.project_mono
#print axioms LBF20Scope.join_project_safe
#print axioms LBF20Scope.project_local
#print axioms LBF20Scope.local_projection_exact
#print axioms LBF20Scope.glue_left
#print axioms LBF20Scope.glue_right
#print axioms LBF20Scope.separator_glue
