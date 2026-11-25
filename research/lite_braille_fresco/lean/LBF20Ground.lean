/-
LBF-20A: weighted ground-term completion.
Exact finite-cost run semantics and conditional saturation theorem.
The finite matrix implementation is audited separately.
Lean 4.19.0, no Mathlib.
-/
namespace LBF20Ground

inductive Term where
  | leaf : Nat → Term
  | un : Nat → Term → Term
  | bin : Nat → Term → Term → Term

abbrev Rules := Term → Term → Prop

inductive Step (R : Rules) : Term → Term → Prop where
  | root {l r : Term} : R l r → Step R l r
  | un (f : Nat) {t u : Term} : Step R t u → Step R (.un f t) (.un f u)
  | binL (f : Nat) {t u : Term} (v : Term) : Step R t u → Step R (.bin f t v) (.bin f u v)
  | binR (f : Nat) (v : Term) {t u : Term} : Step R t u → Step R (.bin f v t) (.bin f v u)

inductive Walk (R : Rules) : Term → Term → Nat → Prop where
  | refl (t : Term) : Walk R t t 0
  | tail {s t u : Term} {n : Nat} : Walk R s t n → Step R t u → Walk R s u (n + 1)

theorem Walk.append {R : Rules} {s t u : Term} {m n : Nat}
    (h : Walk R s t m) (g : Walk R t u n) : Walk R s u (m + n) := by
  induction g with
  | refl => exact h
  | tail _ st ih => exact .tail ih st

theorem Walk.liftUn {R : Rules} {t u : Term} {n : Nat} (h : Walk R t u n) (f : Nat) :
    Walk R (.un f t) (.un f u) n := by
  induction h with
  | refl => exact .refl _
  | tail _ st ih => exact .tail ih (.un f st)

theorem Walk.liftL {R : Rules} {t u : Term} {n : Nat} (h : Walk R t u n) (f : Nat) (v : Term) :
    Walk R (.bin f t v) (.bin f u v) n := by
  induction h with
  | refl => exact .refl _
  | tail _ st ih => exact .tail ih (.binL f v st)

theorem Walk.liftR {R : Rules} {t u : Term} {n : Nat} (h : Walk R t u n) (f : Nat) (v : Term) :
    Walk R (.bin f v t) (.bin f v u) n := by
  induction h with
  | refl => exact .refl _
  | tail _ st ih => exact .tail ih (.binR f v st)

def Within (R : Rules) (s t : Term) (n : Nat) : Prop :=
  ∃ k, k ≤ n ∧ Walk R s t k

theorem Within.trans {R : Rules} {s t u : Term} {m n : Nat}
    (h : Within R s t m) (g : Within R t u n) : Within R s u (m + n) :=
  match h, g with
  | ⟨a, ha, wa⟩, ⟨b, hb, wb⟩ =>
      ⟨a + b, Nat.add_le_add ha hb, wa.append wb⟩

theorem Within.un {R : Rules} {t u : Term} {n : Nat} (h : Within R t u n) (f : Nat) :
    Within R (.un f t) (.un f u) n :=
  match h with | ⟨k, hk, w⟩ => ⟨k, hk, w.liftUn f⟩

theorem Within.bin {R : Rules} {s t u v : Term} {a b : Nat}
    (h : Within R s t a) (g : Within R u v b) (f : Nat) :
    Within R (.bin f s u) (.bin f t v) (a + b) :=
  match h, g with
  | ⟨k, hk, p⟩, ⟨l, hl, q⟩ =>
      ⟨k + l, Nat.add_le_add hk hl, (p.liftL f u).append (q.liftR f t)⟩

/-- Named states are ground terms, with all immediate subterms present. -/
structure SubtermClosed (Q : Term → Prop) : Prop where
  un : ∀ {f t}, Q (.un f t) → Q t
  binL : ∀ {f t u}, Q (.bin f t u) → Q t
  binR : ∀ {f t u}, Q (.bin f t u) → Q u

/-- E q r a means a finite epsilon cost a from source q to structural state r.
    An upper-bound cost is allowed. Root epsilon costs add to child costs. -/
abbrev Eps := Term → Term → Nat → Prop

def Run (Q : Term → Prop) (E : Eps) (q : Term) : Term → Nat → Prop
  | .leaf f, n => Q (.leaf f) ∧ E q (.leaf f) n
  | .un f t, n =>
      ∃ s a b, Q (.un f s) ∧ E q (.un f s) a ∧ Run Q E s t b ∧ a + b = n
  | .bin f t u, n =>
      ∃ s v a b c, Q (.bin f s v) ∧ E q (.bin f s v) a ∧
        Run Q E s t b ∧ Run Q E v u c ∧ a + (b + c) = n

theorem run_sound {R : Rules} {Q : Term → Prop} {E : Eps}
    (hE : ∀ {q r n}, E q r n → Within R q r n) :
    ∀ (t q : Term) (n : Nat), Run Q E q t n → Within R q t n
  | .leaf _, _, _, ⟨_, e⟩ => hE e
  | .un f t, _, _, ⟨s, a, b, _, e, run, eq⟩ =>
      eq ▸ (hE e).trans ((run_sound hE t s b run).un f)
  | .bin f t u, _, _, ⟨s, v, a, b, c, _, e, rt, ru, eq⟩ =>
      eq ▸ (hE e).trans ((run_sound hE t s b rt).bin (run_sound hE u v c ru) f)

theorem run_refl {Q : Term → Prop} {E : Eps} (cl : SubtermClosed Q)
    (hr : ∀ t, Q t → E t t 0) :
    ∀ (t : Term), Q t → Run Q E t t 0
  | .leaf f, h => ⟨h, hr _ h⟩
  | .un f t, h => ⟨t, 0, 0, h, hr _ h, run_refl cl hr t (cl.un h), rfl⟩
  | .bin f t u, h =>
      ⟨t, u, 0, 0, 0, h, hr _ h, run_refl cl hr t (cl.binL h),
        run_refl cl hr u (cl.binR h), rfl⟩

/-- Truncated min-plus closure: composing costs is required only within H. -/
def ClosedEps (E : Eps) (H : Nat) : Prop :=
  ∀ {q r s a b}, E q r a → E r s b → a + b ≤ H → E q s (a + b)

theorem run_extend {Q : Term → Prop} {E : Eps} {H : Nat}
    (cl : ClosedEps E H) {q r : Term} {a : Nat} (e : E q r a) :
    ∀ (t : Term) (b : Nat), a + b ≤ H → Run Q E r t b → Run Q E q t (a + b)
  | .leaf _, _, h, ⟨hq, er⟩ => ⟨hq, cl e er h⟩
  | .un _ _, b, h, ⟨s, c, d, hq, er, rr, eq⟩ => by
      have hc : c ≤ b := eq ▸ Nat.le_add_right c d
      have hb : a + c ≤ H := Nat.le_trans (Nat.add_le_add_left hc a) h
      exact ⟨s, a + c, d, hq, cl e er hb, rr,
        (Nat.add_assoc a c d).trans (congrArg (fun x => a + x) eq)⟩
  | .bin _ _ _, b, h, ⟨s, v, c, d, f, hq, er, rt, ru, eq⟩ => by
      have hc : c ≤ b := eq ▸ Nat.le_add_right c (d + f)
      have hb : a + c ≤ H := Nat.le_trans (Nat.add_le_add_left hc a) h
      exact ⟨s, v, a + c, d, f, hq, cl e er hb, rt, ru,
        (Nat.add_assoc a c (d + f)).trans (congrArg (fun x => a + x) eq)⟩

/-- A sound update inserts the right-hand ground term at cost at most n+1. -/
theorem rule_update_sound {R : Rules} {Q : Term → Prop} {E : Eps}
    (hs : ∀ {q r n}, E q r n → Within R q r n)
    {q l r : Term} {n : Nat} (h : Run Q E q l n) (hr : R l r) :
    Within R q r (n + 1) :=
  match run_sound hs l q n h with
  | ⟨k, hk, p⟩ => ⟨k + 1, Nat.add_le_add_right hk 1, .tail p (.root hr)⟩

/-- Saturation rule: every recognized ground lhs gives its rhs an epsilon edge. -/
def Saturated (R : Rules) (Q : Term → Prop) (E : Eps) (H : Nat) : Prop :=
  ∀ {q l r n}, R l r → Run Q E q l n → n + 1 ≤ H → E q r (n + 1)

theorem run_step {R : Rules} {Q : Term → Prop} {E : Eps} {H : Nat}
    (hq : SubtermClosed Q) (hrefl : ∀ t, Q t → E t t 0)
    (hR : ∀ {l r}, R l r → Q r) (hc : ClosedEps E H) (hs : Saturated R Q E H)
    {t u : Term} (step : Step R t u) :
    ∀ (q : Term) (n : Nat), n + 1 ≤ H → Run Q E q t n → Run Q E q u (n + 1) := by
  induction step with
  | root hr =>
      intro q n hn run
      have edge := hs (q := q) hr run hn
      exact run_extend (Q := Q) (E := E) (H := H) (q := q) (a := n + 1) hc edge _ 0 hn
        (run_refl (Q := Q) (E := E) hq hrefl _ (hR hr))
  | un f st ih =>
      intro q n hn run
      obtain ⟨s, a, b, hq', e, r, eq⟩ := run
      have hb : b ≤ n := eq ▸ Nat.le_add_left b a
      have hb' := Nat.le_trans (Nat.add_le_add_right hb 1) hn
      exact ⟨s, a, b + 1, hq', e, ih s b hb' r, congrArg (fun x => x + 1) eq⟩
  | binL f v st ih =>
      intro q n hn run
      obtain ⟨s, v', a, b, c, hq', e, rt, ru, eq⟩ := run
      have hb : b ≤ n := eq ▸ Nat.le_trans (Nat.le_add_right b c) (Nat.le_add_left (b + c) a)
      have hb' := Nat.le_trans (Nat.add_le_add_right hb 1) hn
      refine ⟨s, v', a, b + 1, c, hq', e, ih s b hb' rt, ru, ?_⟩
      exact (congrArg (fun x => a + x) (Nat.succ_add b c)).trans (congrArg (fun x => x + 1) eq)
  | binR f v st ih =>
      intro q n hn run
      obtain ⟨s, v', a, b, c, hq', e, rt, ru, eq⟩ := run
      have hc' : c ≤ n := eq ▸ Nat.le_trans (Nat.le_add_left c b) (Nat.le_add_left (b + c) a)
      have hc'' := Nat.le_trans (Nat.add_le_add_right hc' 1) hn
      exact ⟨s, v', a, b, c + 1, hq', e, rt, ih v' c hc'' ru, congrArg (fun x => x + 1) eq⟩

theorem walk_complete {R : Rules} {Q : Term → Prop} {E : Eps} {H : Nat}
    (hq : SubtermClosed Q) (hrefl : ∀ t, Q t → E t t 0)
    (hR : ∀ {l r}, R l r → Q r) (hc : ClosedEps E H) (hs : Saturated R Q E H)
    {s t : Term} {n : Nat} (h : Walk R s t n) (hseed : Q s) (hn : n ≤ H) :
    Run Q E s t n := by
  induction h with
  | refl => exact run_refl hq hrefl _ hseed
  | tail path step ih =>
      exact run_step hq hrefl hR hc hs step _ _ hn
        (ih (Nat.le_trans (Nat.le_succ _) hn))

/-- Exact capped reachability, conditional on the finite epsilon table's
    soundness, identities, min-plus closure and rule saturation. -/
theorem capped_exact {R : Rules} {Q : Term → Prop} {E : Eps} {H : Nat}
    (hq : SubtermClosed Q) (hrefl : ∀ t, Q t → E t t 0)
    (hR : ∀ {l r}, R l r → Q r) (hc : ClosedEps E H) (hs : Saturated R Q E H)
    (he : ∀ {q r n}, E q r n → Within R q r n)
    {s t : Term} {n : Nat} (hseed : Q s) (hn : n ≤ H) :
    (∃ k, k ≤ n ∧ Run Q E s t k) ↔ Within R s t n := by
  constructor
  · intro ⟨k, hk, run⟩
    obtain ⟨j, hj, w⟩ := run_sound he t s k run
    exact ⟨j, Nat.le_trans hj hk, w⟩
  · intro ⟨k, hk, w⟩
    exact ⟨k, hk, walk_complete hq hrefl hR hc hs w hseed (Nat.le_trans hk hn)⟩

end LBF20Ground

#print axioms LBF20Ground.Walk.append
#print axioms LBF20Ground.Walk.liftUn
#print axioms LBF20Ground.Walk.liftL
#print axioms LBF20Ground.Walk.liftR
#print axioms LBF20Ground.Within.trans
#print axioms LBF20Ground.Within.un
#print axioms LBF20Ground.Within.bin
#print axioms LBF20Ground.run_sound
#print axioms LBF20Ground.run_refl
#print axioms LBF20Ground.run_extend
#print axioms LBF20Ground.rule_update_sound
#print axioms LBF20Ground.run_step
#print axioms LBF20Ground.walk_complete
#print axioms LBF20Ground.capped_exact
