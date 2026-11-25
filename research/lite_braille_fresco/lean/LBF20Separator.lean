namespace LBF20Separator

abbrev Assignment := Nat → Nat
abbrev Table := Assignment → Prop
abbrev Scope := Nat → Bool

def Agree (A : Scope) (x y : Assignment) : Prop :=
  ∀ i, A i = true → x i = y i

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
  fun i => match A i with
    | true => x i
    | false => y i

theorem glue_left (A : Scope) (x y : Assignment) : Agree A x (Glue A x y) := by
  intro i hi
  exact (congrArg (fun b : Bool => match b with
    | true => x i
    | false => y i) hi).symm

theorem glue_right {A B : Scope} {x y : Assignment}
    (h : ∀ i, A i = true → B i = true → x i = y i) : Agree B y (Glue A x y) := by
  intro i hi
  cases ha : A i with
  | false =>
    exact (congrArg (fun b : Bool => match b with
      | true => x i
      | false => y i) ha).symm
  | true =>
    exact (h i ha hi).symm.trans ((congrArg (fun b : Bool => match b with
      | true => x i
      | false => y i) ha).symm)

theorem separator_glue {A B : Scope} {P Q : Table} {x y : Assignment}
    (hp : Local A P) (hq : Local B Q)
    (px : P x) (qy : Q y)
    (h : ∀ i, A i = true → B i = true → x i = y i) :
    ∃ z, P z ∧ Q z ∧ Agree A x z ∧ Agree B y z :=
  ⟨Glue A x y, hp x _ (glue_left A x y) px,
    hq y _ (glue_right h) qy, glue_left A x y, glue_right h⟩


def ScopeSub (A B : Scope) : Prop :=
  ∀ i, A i = true → B i = true

theorem agree_trans {A : Scope} {x y z : Assignment}
    (h : Agree A x y) (k : Agree A y z) : Agree A x z :=
  fun i hi => (h i hi).trans (k i hi)

theorem project_compose {A B : Scope} {P : Table} (h : ScopeSub A B)
    (x : Assignment) : Project A (Project B P) x ↔ Project A P x := by
  constructor
  · intro ⟨y, ⟨z, hz, hyz⟩, hxy⟩
    exact ⟨z, hz, agree_trans hxy (fun i hi => hyz i (h i hi))⟩
  · intro ⟨y, hy, hxy⟩
    exact ⟨y, project_contains B P y hy, hxy⟩

/-- Exact separator message: the other side can be replaced by its
    projection onto the shared variables when querying the first bag. -/
theorem separator_message_exact {A B S : Scope} {P Q : Table}
    (hp : Local A P) (hq : Local B Q) (hs : ScopeSub S A)
    (cover : ∀ i, A i = true → B i = true → S i = true)
    (x : Assignment) :
    Project A (Join P Q) x ↔ P x ∧ Project S Q x := by
  constructor
  · intro ⟨y, hy, hxy⟩
    exact ⟨hp y x (agree_symm hxy) hy.1,
      ⟨y, hy.2, fun i hi => hxy i (hs i hi)⟩⟩
  · intro ⟨px, y, qy, hxy⟩
    obtain ⟨z, pz, qz, hxz, hyz⟩ :=
      separator_glue hp hq px qy (fun i ha hb => hxy i (cover i ha hb))
    exact ⟨z, ⟨pz, qz⟩, hxz⟩

/-- A query confined to a calibrated bag can be answered by projection. -/
theorem cached_query_exact {A B : Scope} {P : Table} (h : ScopeSub A B)
    (x : Assignment) : Project A (Project B P) x ↔ Project A P x :=
  project_compose h x

/-- Combining local evidence with a calibrated bag is exact for any
    evidence depending only on that bag's variables. -/
theorem local_evidence_exact {A : Scope} {P E : Table}
    (he : Local A E) (x : Assignment) :
    Project A (Join P E) x ↔ Project A P x ∧ E x := by
  constructor
  · intro ⟨y, hy, hxy⟩
    exact ⟨⟨y, hy.1, hxy⟩, he y x (agree_symm hxy) hy.2⟩
  · intro ⟨⟨y, py, hxy⟩, ex⟩
    exact ⟨y, ⟨py, he x y hxy ex⟩, hxy⟩

end LBF20Separator
#print axioms LBF20Separator.agree_trans
#print axioms LBF20Separator.project_compose
#print axioms LBF20Separator.separator_message_exact
#print axioms LBF20Separator.cached_query_exact
#print axioms LBF20Separator.local_evidence_exact
