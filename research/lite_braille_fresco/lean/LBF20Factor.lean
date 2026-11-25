namespace LBF20Factor

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


/-- Residual components with disjoint scopes can be glued independently. -/
theorem disjoint_join_feasible {A B : Scope} {P Q : Table}
    (hp : Local A P) (hq : Local B Q)
    (hd : ∀ i, A i = true → B i = true → False) :
    (∃ x, Join P Q x) ↔ (∃ x, P x) ∧ (∃ y, Q y) := by
  constructor
  · intro ⟨x, px, qx⟩
    exact ⟨⟨x,px⟩,⟨x,qx⟩⟩
  · intro ⟨⟨x,px⟩,⟨y,qy⟩⟩
    obtain ⟨z,pz,qz,_,_⟩ :=
      separator_glue hp hq px qy (fun i ha hb => False.elim (hd i ha hb))
    exact ⟨z,pz,qz⟩

theorem disjoint_support_exact {A B : Scope} {P Q : Table}
    (hp : Local A P) (hq : Local B Q)
    (hd : ∀ i, A i = true → B i = true → False) (x : Assignment) :
    Project A (Join P Q) x ↔ P x ∧ (∃ y, Q y) := by
  constructor
  · intro ⟨y,hy,he⟩
    exact ⟨hp y x (agree_symm he) hy.1,⟨y,hy.2⟩⟩
  · intro ⟨px,⟨y,qy⟩⟩
    obtain ⟨z,pz,qz,hxz,_⟩ :=
      separator_glue hp hq px qy (fun i ha hb => False.elim (hd i ha hb))
    exact ⟨z,⟨pz,qz⟩,hxz⟩

/-- A deterministic layered diagram can encode its layer in K. -/
def Run {K : Type} (step : K → Nat → Option K) (start : K) :
    List Nat → Option K
  | [] => some start
  | a :: rest =>
    match step start a with
    | none => none
    | some next => Run step next rest

theorem run_append {K : Type} (step : K → Nat → Option K)
    (xs ys : List Nat) (start : K) :
    Run step start (xs ++ ys) =
      (match Run step start xs with
       | none => none
       | some t => Run step t ys) := by
  induction xs generalizing start with
  | nil => rfl
  | cons a xs ih =>
    change (match step start a with
      | none => none
      | some t => Run step t (xs ++ ys)) =
      (match (match step start a with
        | none => none
        | some t => Run step t xs) with
       | none => none
       | some t => Run step t ys)
    cases step start a with
    | none => rfl
    | some t => exact ih t

theorem suffix_congruence {K : Type} (step : K → Nat → Option K)
    {a b : K} (h : a = b) (suffix : List Nat) :
    Run step a suffix = Run step b suffix :=
  congrArg (fun k => Run step k suffix) h

/-- An edge has accepting support exactly when its forward prefix and
    backward suffix can be glued at the same states. -/
theorem accepting_edge {K : Type} (step : K → Nat → Option K)
    (start : K) (xs ys : List Nat) (a : Nat) (ok : K → Prop) :
    (∃ t, Run step start (xs ++ (a :: ys)) = some t ∧ ok t) ↔
    (∃ u v t, Run step start xs = some u ∧ step u a = some v ∧
      Run step v ys = some t ∧ ok t) := by
  constructor
  · intro ⟨t, ht, ho⟩
    cases hu : Run step start xs with
    | none =>
      rw [run_append, hu] at ht
      cases ht
    | some u =>
      rw [run_append, hu] at ht
      change Run step u (a :: ys) = some t at ht
      cases hv : step u a with
      | none =>
        change (match step u a with | none => none | some z => Run step z ys) = some t at ht
        rw [hv] at ht
        cases ht
      | some v =>
        rw [Run, hv] at ht
        exact ⟨u,v,t,rfl,hv,ht,ho⟩
  · intro ⟨u,v,t,hu,hv,ht,ho⟩
    refine ⟨t, ?_, ho⟩
    rw [run_append, hu]
    change Run step u (a :: ys) = some t
    change (match step u a with | none => none | some z => Run step z ys) = some t
    rw [hv]
    exact ht

/-- The terminal predicate must test every component in that same branch. -/
theorem terminal_component_gate {K C : Type} (reachable : K → Prop)
    (feasible : K → C → Prop) (terminalSupport : K → Prop) :
    (∃ k, reachable k ∧ (∀ c, feasible k c) ∧ terminalSupport k) →
    (∀ c, ∃ k, reachable k ∧ feasible k c) := by
  intro ⟨k,hr,hf,_⟩ c
  exact ⟨k,hr,hf c⟩

end LBF20Factor
#print axioms LBF20Factor.disjoint_join_feasible
#print axioms LBF20Factor.disjoint_support_exact
#print axioms LBF20Factor.run_append
#print axioms LBF20Factor.suffix_congruence
#print axioms LBF20Factor.accepting_edge
#print axioms LBF20Factor.terminal_component_gate
