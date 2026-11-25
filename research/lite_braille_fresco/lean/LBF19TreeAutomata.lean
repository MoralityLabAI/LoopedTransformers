/-
  LBF-19 Track T: tree-automata abstraction of variable-size expression trees.
  Lean 4.19.0, core only, zero axioms.
-/
namespace LBF19TreeAutomata

inductive Term : Type where
  | leaf : Nat → Term
  | un   : Nat → Term → Term
  | bin  : Nat → Term → Term → Term

inductive Pat : Type where
  | var  : Nat → Pat
  | leaf : Nat → Pat
  | un   : Nat → Pat → Pat
  | bin  : Nat → Pat → Pat → Pat

def Pat.subst (σ : Nat → Term) : Pat → Term
  | .var x => σ x
  | .leaf f => .leaf f
  | .un f p => .un f (p.subst σ)
  | .bin f p q => .bin f (p.subst σ) (q.subst σ)

def Pat.occ (x : Nat) : Pat → Bool
  | .var y => Nat.beq x y
  | .leaf _ => false
  | .un _ p => p.occ x
  | .bin _ p q => p.occ x || q.occ x

/-- left-linearity: every variable occurs at most once -/
def Pat.Linear : Pat → Prop
  | .var _ => True
  | .leaf _ => True
  | .un _ p => p.Linear
  | .bin _ p q => p.Linear ∧ q.Linear ∧ ∀ x, p.occ x = true → q.occ x = false

theorem beq_self_true : ∀ n : Nat, Nat.beq n n = true
  | 0 => rfl
  | n + 1 => beq_self_true n

theorem occ_var_self (x : Nat) : (Pat.var x).occ x = true := beq_self_true x

theorem occ_bin_l {f : Nat} {p r : Pat} {x : Nat} (h : p.occ x = true) : (Pat.bin f p r).occ x = true := by
  show (p.occ x || r.occ x) = true
  rw [h]; rfl

theorem occ_bin_r {f : Nat} {p r : Pat} {x : Nat} (h : r.occ x = true) : (Pat.bin f p r).occ x = true := by
  show (p.occ x || r.occ x) = true
  rw [h]; cases p.occ x <;> rfl

theorem occ_bin_inv {f : Nat} {p r : Pat} {x : Nat} (h : (Pat.bin f p r).occ x = true) :
    p.occ x = true ∨ r.occ x = true := by
  have h' : (p.occ x || r.occ x) = true := h
  cases h1 : p.occ x with
  | true => exact Or.inl rfl
  | false =>
    cases h2 : r.occ x with
    | true => exact Or.inr rfl
    | false => rw [h1, h2] at h'; exact absurd h' (by decide)

structure Rule where
  lhs : Pat
  rhs : Pat

def Rule.Ok (r : Rule) : Prop :=
  r.lhs.Linear ∧ ∀ x, r.rhs.occ x = true → r.lhs.occ x = true

/-- one-step rewriting at any position, with rule set R (a predicate) -/
inductive Step (R : Rule → Prop) : Term → Term → Prop where
  | root (r : Rule) (hr : R r) (σ : Nat → Term) :
      Step R (r.lhs.subst σ) (r.rhs.subst σ)
  | un (f : Nat) {t t' : Term} : Step R t t' → Step R (.un f t) (.un f t')
  | binL (f : Nat) {t t' : Term} (u : Term) : Step R t t' → Step R (.bin f t u) (.bin f t' u)
  | binR (f : Nat) (t : Term) {u u' : Term} : Step R u u' → Step R (.bin f t u) (.bin f t u')

/-- k-bounded reachability from a set S -/
def ReachN (R : Rule → Prop) (S : Term → Prop) : Nat → Term → Prop
  | 0, t => S t
  | k + 1, t => ReachN R S k t ∨ ∃ t', ReachN R S k t' ∧ Step R t' t

-- ---------------------------------------------------------------------------
-- Nondeterministic bottom-up tree automata with epsilon transitions (relational)
-- ---------------------------------------------------------------------------
structure NTA where
  leaf  : Nat → Nat → Prop
  un    : Nat → Nat → Nat → Prop        -- sym, child state, parent state
  bin   : Nat → Nat → Nat → Nat → Prop  -- sym, left, right, parent
  eps   : Nat → Nat → Prop
  final : Nat → Prop

inductive EpsStar (A : NTA) : Nat → Nat → Prop where
  | refl (q : Nat) : EpsStar A q q
  | tail {a b c : Nat} : EpsStar A a b → A.eps b c → EpsStar A a c

theorem EpsStar.trans {A : NTA} {a b c : Nat} (h1 : EpsStar A a b) (h2 : EpsStar A b c) :
    EpsStar A a c := by
  induction h2 with
  | refl => exact h1
  | tail _ h' ih => exact .tail ih h'

/-- Run q t : the automaton can evaluate t to state q -/
def Run (A : NTA) : Nat → Term → Prop
  | q, .leaf f => ∃ q0, EpsStar A q0 q ∧ A.leaf f q0
  | q, .un f t => ∃ q0, EpsStar A q0 q ∧ ∃ q1, A.un f q1 q0 ∧ Run A q1 t
  | q, .bin f t u => ∃ q0, EpsStar A q0 q ∧ ∃ q1 q2, A.bin f q1 q2 q0 ∧ Run A q1 t ∧ Run A q2 u

def Accepts (A : NTA) (t : Term) : Prop := ∃ q, A.final q ∧ Run A q t

/-- epsilon-extension of a run -/
theorem Run.ext {A : NTA} {q q' : Nat} (h : EpsStar A q q') : ∀ t, Run A q t → Run A q' t
  | .leaf _, ⟨q0, h0, h1⟩ => ⟨q0, EpsStar.trans h0 h, h1⟩
  | .un _ _, ⟨q0, h0, r⟩ => ⟨q0, EpsStar.trans h0 h, r⟩
  | .bin _ _ _, ⟨q0, h0, r⟩ => ⟨q0, EpsStar.trans h0 h, r⟩

/-- automaton inclusion (transition-set inclusion; the support order) -/
structure Incl (A B : NTA) : Prop where
  leaf  : ∀ f q, A.leaf f q → B.leaf f q
  un    : ∀ f a q, A.un f a q → B.un f a q
  bin   : ∀ f a b q, A.bin f a b q → B.bin f a b q
  eps   : ∀ a b, A.eps a b → B.eps a b
  final : ∀ q, A.final q → B.final q

theorem EpsStar.mono {A B : NTA} (h : Incl A B) {a b : Nat} (e : EpsStar A a b) : EpsStar B a b := by
  induction e with
  | refl => exact .refl _
  | tail _ he ih => exact .tail ih (h.eps _ _ he)

/-- (c0) monotonicity of the language in the support -/
theorem Run.mono {A B : NTA} (h : Incl A B) : ∀ (t : Term) (q : Nat), Run A q t → Run B q t
  | .leaf _, _, ⟨q0, h0, h1⟩ => ⟨q0, EpsStar.mono h h0, h.leaf _ _ h1⟩
  | .un _ t, _, ⟨q0, h0, q1, h1, r⟩ =>
      ⟨q0, EpsStar.mono h h0, q1, h.un _ _ _ h1, Run.mono h t _ r⟩
  | .bin _ t u, _, ⟨q0, h0, q1, q2, h1, r1, r2⟩ =>
      ⟨q0, EpsStar.mono h h0, q1, q2, h.bin _ _ _ _ h1, Run.mono h t _ r1, Run.mono h u _ r2⟩

theorem Accepts.mono {A B : NTA} (h : Incl A B) {t : Term} (a : Accepts A t) : Accepts B t :=
  match a with
  | ⟨q, hf, r⟩ => ⟨q, h.final _ hf, Run.mono h t q r⟩

-- ---------------------------------------------------------------------------
-- Pattern matching in an automaton
-- ---------------------------------------------------------------------------
/-- lhs matching: variable v is read at the exact state ρ v of the position. -/
def MatchL (A : NTA) (ρ : Nat → Nat) : Pat → Nat → Prop
  | .var x, q => ρ x = q
  | .leaf f, q => ∃ q0, EpsStar A q0 q ∧ A.leaf f q0
  | .un f p, q => ∃ q0, EpsStar A q0 q ∧ ∃ q1, A.un f q1 q0 ∧ MatchL A ρ p q1
  | .bin f p r, q => ∃ q0, EpsStar A q0 q ∧ ∃ q1 q2, A.bin f q1 q2 q0 ∧
      MatchL A ρ p q1 ∧ MatchL A ρ r q2

/-- rhs construction: a variable may reach q through epsilon steps. -/
def MatchR (A : NTA) (ρ : Nat → Nat) : Pat → Nat → Prop
  | .var x, q => EpsStar A (ρ x) q
  | .leaf f, q => ∃ q0, EpsStar A q0 q ∧ A.leaf f q0
  | .un f p, q => ∃ q0, EpsStar A q0 q ∧ ∃ q1, A.un f q1 q0 ∧ MatchR A ρ p q1
  | .bin f p r, q => ∃ q0, EpsStar A q0 q ∧ ∃ q1 q2, A.bin f q1 q2 q0 ∧
      MatchR A ρ p q1 ∧ MatchR A ρ r q2

/-- Matched rhs (with substitution consistent with ρ) is accepted. -/
theorem matchR_run {A : NTA} (ρ : Nat → Nat) (σ : Nat → Term) :
    ∀ (p : Pat) (q : Nat), MatchR A ρ p q →
      (∀ x, p.occ x = true → Run A (ρ x) (σ x)) → Run A q (p.subst σ)
  | .var x, q, h, hv => by
      exact Run.ext h _ (hv x (occ_var_self x))
  | .leaf _, _, h, _ => h
  | .un f p, q, ⟨q0, h0, q1, h1, m⟩, hv =>
      ⟨q0, h0, q1, h1, matchR_run ρ σ p q1 m (fun x hx => hv x (by first | exact hx | exact occ_bin_l hx | exact occ_bin_r hx))⟩
  | .bin f p r, q, ⟨q0, h0, q1, q2, h1, m1, m2⟩, hv =>
      ⟨q0, h0, q1, q2, h1,
        matchR_run ρ σ p q1 m1 (fun x hx => hv x (by first | exact hx | exact occ_bin_l hx | exact occ_bin_r hx)),
        matchR_run ρ σ r q2 m2 (fun x hx => hv x (by first | exact hx | exact occ_bin_l hx | exact occ_bin_r hx))⟩

/-- MatchL depends only on ρ at the variables of the pattern. -/
theorem matchL_congr {A : NTA} {ρ ρ' : Nat → Nat} :
    ∀ (p : Pat) (q : Nat), (∀ x, p.occ x = true → ρ x = ρ' x) → MatchL A ρ p q → MatchL A ρ' p q
  | .var x, q, h, m => Eq.trans (Eq.symm (h x (occ_var_self x))) m
  | .leaf _, _, _, m => m
  | .un f p, q, h, ⟨q0, h0, q1, h1, m⟩ =>
      ⟨q0, h0, q1, h1, matchL_congr p q1 (fun x hx => h x hx) m⟩
  | .bin f p r, q, h, ⟨q0, h0, q1, q2, h1, m1, m2⟩ =>
      ⟨q0, h0, q1, q2, h1,
        matchL_congr p q1 (fun x hx => h x (occ_bin_l hx)) m1,
        matchL_congr r q2 (fun x hx => h x (occ_bin_r hx)) m2⟩

/-- Completeness of lhs matching for left-linear patterns. -/
theorem matchL_complete {A : NTA} (σ : Nat → Term) :
    ∀ (p : Pat) (q : Nat), p.Linear → Run A q (p.subst σ) →
      ∃ ρ : Nat → Nat, MatchL A ρ p q ∧ ∀ x, p.occ x = true → Run A (ρ x) (σ x)
  | .var x, q, _, r => ⟨fun _ => q, rfl, fun y hy => by
      have e : y = x := Nat.eq_of_beq_eq_true hy
      subst e; exact r⟩
  | .leaf _, q, _, r => ⟨fun _ => q, r, fun y hy => nomatch hy⟩
  | .un f p, q, hl, ⟨q0, h0, q1, h1, r⟩ =>
      match matchL_complete σ p q1 hl r with
      | ⟨ρ, m, hr⟩ => ⟨ρ, ⟨q0, h0, q1, h1, m⟩, fun x hx => hr x hx⟩
  | .bin f p s, q, ⟨hl1, hl2, hd⟩, ⟨q0, h0, q1, q2, h1, r1, r2⟩ =>
      match matchL_complete σ p q1 hl1 r1, matchL_complete σ s q2 hl2 r2 with
      | ⟨ρ1, m1, hr1⟩, ⟨ρ2, m2, hr2⟩ =>
        have c1 : MatchL A (fun x => cond (p.occ x) (ρ1 x) (ρ2 x)) p q1 :=
          matchL_congr p q1 (fun x hx => by
            show ρ1 x = cond (p.occ x) (ρ1 x) (ρ2 x)
            rw [hx]; rfl) m1
        have c2 : MatchL A (fun x => cond (p.occ x) (ρ1 x) (ρ2 x)) s q2 :=
          matchL_congr s q2 (fun x hx => by
            have hpf : p.occ x = false := by
              cases h : p.occ x with
              | false => rfl
              | true => have := hd x h; rw [this] at hx; exact absurd hx (by decide)
            show ρ2 x = cond (p.occ x) (ρ1 x) (ρ2 x)
            rw [hpf]; rfl) m2
        ⟨fun x => cond (p.occ x) (ρ1 x) (ρ2 x), ⟨q0, h0, q1, q2, h1, c1, c2⟩, fun x hx => by
          show Run A (cond (p.occ x) (ρ1 x) (ρ2 x)) (σ x)
          cases hp : p.occ x with
          | true => exact hr1 x hp
          | false =>
            have hs : s.occ x = true :=
              match occ_bin_inv hx with
              | Or.inl h => by rw [hp] at h; exact absurd h (by decide)
              | Or.inr h => h
            exact hr2 x hs⟩

-- ---------------------------------------------------------------------------
-- Completion step and soundness
-- ---------------------------------------------------------------------------
/-- A' is a one-round completion of A for rule set R. -/
structure Complete1 (R : Rule → Prop) (A A' : NTA) : Prop where
  incl : Incl A A'
  ok   : ∀ r, R r → r.Ok
  step : ∀ r, R r → ∀ ρ q, MatchL A ρ r.lhs q → MatchR A' ρ r.rhs q

/-- (a) one-round: every rewrite step from a term evaluating to q yields a term evaluating to q. -/
theorem complete1_run_step {R : Rule → Prop} {A A' : NTA} (h : Complete1 R A A') :
    ∀ {t t' : Term}, Step R t t' → ∀ q, Run A q t → Run A' q t'
  | _, _, .root r hr σ, q, run => by
      have hok := h.ok r hr
      match matchL_complete σ r.lhs q hok.1 run with
      | ⟨ρ, m, hv⟩ =>
        exact matchR_run ρ σ r.rhs q (h.step r hr ρ q m)
          (fun x hx => Run.mono h.incl _ _ (hv x (hok.2 x hx)))
  | _, _, .un f s, q, ⟨q0, h0, q1, h1, r⟩ =>
      ⟨q0, EpsStar.mono h.incl h0, q1, h.incl.un _ _ _ h1, complete1_run_step h s q1 r⟩
  | _, _, .binL f u s, q, ⟨q0, h0, q1, q2, h1, r1, r2⟩ =>
      ⟨q0, EpsStar.mono h.incl h0, q1, q2, h.incl.bin _ _ _ _ h1,
        complete1_run_step h s q1 r1, Run.mono h.incl _ _ r2⟩
  | _, _, .binR f t s, q, ⟨q0, h0, q1, q2, h1, r1, r2⟩ =>
      ⟨q0, EpsStar.mono h.incl h0, q1, q2, h.incl.bin _ _ _ _ h1,
        Run.mono h.incl _ _ r1, complete1_run_step h s q2 r2⟩

theorem complete1_accepts_step {R : Rule → Prop} {A A' : NTA} (h : Complete1 R A A')
    {t t' : Term} (s : Step R t t') (a : Accepts A t) : Accepts A' t' :=
  match a with
  | ⟨q, hf, r⟩ => ⟨q, h.incl.final _ hf, complete1_run_step h s q r⟩

/-- Language-level soundness of one abstract round (completion and/or widening). -/
def RoundSound (R : Rule → Prop) (A A' : NTA) : Prop :=
  (∀ t, Accepts A t → Accepts A' t) ∧ (∀ t t', Step R t t' → Accepts A t → Accepts A' t')

theorem complete1_roundSound {R : Rule → Prop} {A A' : NTA} (h : Complete1 R A A') :
    RoundSound R A A' :=
  ⟨fun _ a => Accepts.mono h.incl a, fun _ _ s a => complete1_accepts_step h s a⟩

/-- (b) k-step soundness: any chain of sound rounds over-approximates k-bounded reachability. -/
theorem chain_sound (R : Rule → Prop) (A0 : NTA) (seq : Nat → NTA)
    (h0 : ∀ t, Accepts A0 t → Accepts (seq 0) t)
    (hs : ∀ i, RoundSound R (seq i) (seq (i + 1))) :
    ∀ (k : Nat) (t : Term), ReachN R (Accepts A0) k t → Accepts (seq k) t
  | 0, t, h => h0 t h
  | k + 1, t, Or.inl h => (hs k).1 t (chain_sound R A0 seq h0 hs k t h)
  | k + 1, t, Or.inr ⟨t', ht', st⟩ =>
      (hs k).2 t' t st (chain_sound R A0 seq h0 hs k t' ht')

/-- T_R^k iterated: a sequence of one-round completions (witnessed). -/
theorem completion_chain_sound (R : Rule → Prop) (A0 : NTA) (seq : Nat → NTA)
    (h0 : seq 0 = A0)
    (hs : ∀ i, Complete1 R (seq i) (seq (i + 1))) :
    ∀ (k : Nat) (t : Term), ReachN R (Accepts A0) k t → Accepts (seq k) t :=
  chain_sound R A0 seq (fun t a => by rw [h0]; exact a)
    (fun i => complete1_roundSound (hs i))

-- ---------------------------------------------------------------------------
-- (c) Widening: state merging
-- ---------------------------------------------------------------------------
def NTA.map (h : Nat → Nat) (A : NTA) : NTA where
  leaf f q' := ∃ q, A.leaf f q ∧ h q = q'
  un f a' q' := ∃ a q, A.un f a q ∧ h a = a' ∧ h q = q'
  bin f a' b' q' := ∃ a b q, A.bin f a b q ∧ h a = a' ∧ h b = b' ∧ h q = q'
  eps a' b' := ∃ a b, A.eps a b ∧ h a = a' ∧ h b = b'
  final q' := ∃ q, A.final q ∧ h q = q'

theorem EpsStar.map (h : Nat → Nat) {A : NTA} {a b : Nat} (e : EpsStar A a b) :
    EpsStar (A.map h) (h a) (h b) := by
  induction e with
  | refl => exact .refl _
  | tail _ he ih => exact .tail ih ⟨_, _, he, rfl, rfl⟩

/-- merging states maps runs to runs: L_q(A) ⊆ L_{h q}(A/h). -/
theorem Run.map (h : Nat → Nat) {A : NTA} :
    ∀ (t : Term) (q : Nat), Run A q t → Run (A.map h) (h q) t
  | .leaf _, _, ⟨q0, h0, h1⟩ => ⟨h q0, EpsStar.map h h0, q0, h1, rfl⟩
  | .un _ t, _, ⟨q0, h0, q1, h1, r⟩ =>
      ⟨h q0, EpsStar.map h h0, h q1, ⟨q1, q0, h1, rfl, rfl⟩, Run.map h t _ r⟩
  | .bin _ t u, _, ⟨q0, h0, q1, q2, h1, r1, r2⟩ =>
      ⟨h q0, EpsStar.map h h0, h q1, h q2, ⟨q1, q2, q0, h1, rfl, rfl, rfl⟩,
        Run.map h t _ r1, Run.map h u _ r2⟩

theorem widen_accepts (h : Nat → Nat) {A : NTA} {t : Term} (a : Accepts A t) :
    Accepts (A.map h) t :=
  match a with
  | ⟨q, hf, r⟩ => ⟨h q, ⟨q, hf, rfl⟩, Run.map h t q r⟩

/-- (c) complete-then-widen is a sound round. -/
theorem complete_widen_roundSound {R : Rule → Prop} {A A' : NTA} (h : Complete1 R A A')
    (m : Nat → Nat) : RoundSound R A (A'.map m) :=
  ⟨fun _ a => widen_accepts m (Accepts.mono h.incl a),
   fun _ _ s a => widen_accepts m (complete1_accepts_step h s a)⟩

theorem completion_widening_chain_sound (R : Rule → Prop) (A0 : NTA) (seq : Nat → NTA)
    (ms : Nat → (Nat → Nat)) (mid : Nat → NTA)
    (h0 : seq 0 = A0)
    (hc : ∀ i, Complete1 R (seq i) (mid i))
    (hw : ∀ i, seq (i + 1) = (mid i).map (ms i)) :
    ∀ (k : Nat) (t : Term), ReachN R (Accepts A0) k t → Accepts (seq k) t :=
  chain_sound R A0 seq (fun t a => by rw [h0]; exact a)
    (fun i => by rw [hw i]; exact complete_widen_roundSound (hc i) (ms i))

-- ---------------------------------------------------------------------------
-- (d) Unary chains = strings
-- ---------------------------------------------------------------------------
/-- word w followed by tail term t -/
def wrap : List Nat → Term → Term
  | [], t => t
  | c :: w, t => .un c (wrap w t)

def encode (w : List Nat) : Term := wrap w (.leaf 0)

def wrapPat : List Nat → Pat → Pat
  | [], p => p
  | c :: w, p => .un c (wrapPat w p)

/-- string rule u → v as tree rule with tail variable 0 -/
def stringRule (u v : List Nat) : Rule := ⟨wrapPat u (.var 0), wrapPat v (.var 0)⟩

theorem wrap_append (u w : List Nat) (t : Term) : wrap (u ++ w) t = wrap u (wrap w t) :=
  match u with
  | [] => rfl
  | c :: u' => congrArg (Term.un c) (wrap_append u' w t)

theorem subst_wrapPat (σ : Nat → Term) (u : List Nat) :
    (wrapPat u (.var 0)).subst σ = wrap u (σ 0) :=
  match u with
  | [] => rfl
  | c :: u' => congrArg (Term.un c) (subst_wrapPat σ u')

theorem step_wrap {R : Rule → Prop} {t t' : Term} (s : Step R t t') :
    ∀ pre : List Nat, Step R (wrap pre t) (wrap pre t')
  | [] => s
  | c :: pre' => .un c (step_wrap s pre')

/-- every string rewrite pre·u·suf → pre·v·suf is one tree step in the unary chain. -/
theorem string_step_is_tree_step {R : Rule → Prop} (u v pre suf : List Nat)
    (hr : R (stringRule u v)) :
    Step R (encode (pre ++ u ++ suf)) (encode (pre ++ v ++ suf)) := by
  have e1 : encode (pre ++ u ++ suf) = wrap pre (wrap u (encode suf)) := by
    unfold encode; rw [wrap_append, wrap_append]
  have e2 : encode (pre ++ v ++ suf) = wrap pre (wrap v (encode suf)) := by
    unfold encode; rw [wrap_append, wrap_append]
  rw [e1, e2]
  have h : Step R ((wrapPat u (.var 0)).subst (fun _ => encode suf))
      ((wrapPat v (.var 0)).subst (fun _ => encode suf)) :=
    Step.root (stringRule u v) hr (fun _ => encode suf)
  rw [subst_wrapPat, subst_wrapPat] at h
  exact step_wrap h pre

theorem stringRule_ok (u v : List Nat) : (stringRule u v).Ok := by
  have lin : ∀ w : List Nat, (wrapPat w (.var 0)).Linear := by
    intro w; induction w with
    | nil => exact True.intro
    | cons c w ih => exact ih
  have occ : ∀ (w : List Nat) x, (wrapPat w (.var 0)).occ x = Nat.beq x 0 := by
    intro w x; induction w with
    | nil => rfl
    | cons c w ih => exact ih
  refine ⟨lin u, fun x hx => ?_⟩
  have hx' : (wrapPat v (.var 0)).occ x = true := hx
  rw [occ] at hx'
  show (wrapPat u (.var 0)).occ x = true
  rw [occ]; exact hx'

end LBF19TreeAutomata

open LBF19TreeAutomata in
#print axioms EpsStar.trans
open LBF19TreeAutomata in
#print axioms Run.ext
open LBF19TreeAutomata in
#print axioms Run.mono
open LBF19TreeAutomata in
#print axioms Accepts.mono
open LBF19TreeAutomata in
#print axioms matchR_run
open LBF19TreeAutomata in
#print axioms matchL_congr
open LBF19TreeAutomata in
#print axioms matchL_complete
open LBF19TreeAutomata in
#print axioms complete1_run_step
open LBF19TreeAutomata in
#print axioms complete1_accepts_step
open LBF19TreeAutomata in
#print axioms complete1_roundSound
open LBF19TreeAutomata in
#print axioms chain_sound
open LBF19TreeAutomata in
#print axioms completion_chain_sound
open LBF19TreeAutomata in
#print axioms Run.map
open LBF19TreeAutomata in
#print axioms widen_accepts
open LBF19TreeAutomata in
#print axioms complete_widen_roundSound
open LBF19TreeAutomata in
#print axioms completion_widening_chain_sound
open LBF19TreeAutomata in
#print axioms string_step_is_tree_step
open LBF19TreeAutomata in
#print axioms stringRule_ok
