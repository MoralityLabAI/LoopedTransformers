/-
  LBF-18: Incremental Relational Fresco, Cache Soundness, 2x2 Bank Interaction,
  and Provenance Collision Thresholds.
  Standalone Lean 4.19.0 formalization (zero external dependencies, zero axioms).
-/

namespace LBF18IncrementalAndCollisions

inductive Sym : Type where
  | a | b | c | d | e | f
  deriving DecidableEq, Repr

abbrev WordL : Type := List Sym

-- ============================================================================
-- Section 1. Incremental Frontier Factorization & Keyed Cache Soundness
-- ============================================================================

structure CacheEntry (K V : Type) where
  key : K
  val : V

def lookupCache {K V : Type} [DecidableEq K] : List (CacheEntry K V) → K → Option V
  | [], _ => none
  | e :: rest, k => if e.key = k then some e.val else lookupCache rest k

def CacheSound {K V : Type} (eval : K → V) : List (CacheEntry K V) → Prop
  | [] => True
  | e :: rest => e.val = eval e.key ∧ CacheSound eval rest

def evalIncremental {K V : Type} [DecidableEq K]
    (eval : K → V) (cache : List (CacheEntry K V)) (k : K) : V :=
  match lookupCache cache k with
  | some v => v
  | none => eval k

/-- Theorem 1: Soundness of keyed incremental evaluation on hit and miss. -/
theorem keyed_cache_lookup_sound {K V : Type} [DecidableEq K]
    (eval : K → V) :
    ∀ (cache : List (CacheEntry K V)), CacheSound eval cache →
      ∀ (k : K), evalIncremental eval cache k = eval k
  | [], _, _ => rfl
  | e :: rest, ⟨he, hrest⟩, k =>
      match decEq e.key k with
      | isTrue heq =>
          have h_if : (if e.key = k then some e.val else lookupCache rest k) = some e.val :=
            if_pos heq
          have h_step : evalIncremental eval (e :: rest) k = e.val :=
            congrArg (fun o => match o with | some v => v | none => eval k) h_if
          have h_eval : eval e.key = eval k := congrArg eval heq
          Eq.trans h_step (Eq.trans he h_eval)
      | isFalse hne =>
          have h_if : (if e.key = k then some e.val else lookupCache rest k) = lookupCache rest k :=
            if_neg hne
          have h_step : evalIncremental eval (e :: rest) k = evalIncremental eval rest k :=
            congrArg (fun o => match o with | some v => v | none => eval k) h_if
          Eq.trans h_step (keyed_cache_lookup_sound eval rest hrest k)

structure WordDepKey where
  checkpointId : Nat
  rulesId      : Nat
  envModeId    : Nat
  passes       : Nat
  word         : WordL
  deriving DecidableEq

/-- Theorem 2: Fail-closed cache rejection whenever any dependency changes. -/
theorem keyed_cache_rejects_mismatch {V : Type}
    (k₀ k₁ : WordDepKey) (v₀ : V) (hne : k₀ ≠ k₁) :
    lookupCache [⟨k₀, v₀⟩] k₁ = none :=
  if_neg hne

/-- Theorem 3: Incremental frontier word-envelope map equals uncached batch map. -/
theorem incremental_word_map_equiv {V : Type}
    (evalWord : WordDepKey → V)
    (cache : List (CacheEntry WordDepKey V))
    (hSound : CacheSound evalWord cache) :
    ∀ (keys : List WordDepKey),
      keys.map (evalIncremental evalWord cache) = keys.map evalWord
  | [] => rfl
  | k :: ks =>
      have hd := keyed_cache_lookup_sound evalWord cache hSound k
      have tl := incremental_word_map_equiv evalWord cache hSound ks
      Eq.trans
        (congrArg (fun x => x :: ks.map (evalIncremental evalWord cache)) hd)
        (congrArg (List.cons (evalWord k)) tl)

structure PairDepKey where
  checkpointId : Nat
  rulesId      : Nat
  envModeId    : Nat
  passes       : Nat
  uWord        : WordL
  vWord        : WordL
  deriving DecidableEq

/-- Theorem 4: Incremental static pairwise ribbon map equals uncached pairwise map. -/
theorem incremental_pair_map_equiv {V : Type}
    (evalPair : PairDepKey → V)
    (cache : List (CacheEntry PairDepKey V))
    (hSound : CacheSound evalPair cache) :
    ∀ (pairs : List PairDepKey),
      pairs.map (evalIncremental evalPair cache) = pairs.map evalPair
  | [] => rfl
  | p :: ps =>
      have hd := keyed_cache_lookup_sound evalPair cache hSound p
      have tl := incremental_pair_map_equiv evalPair cache hSound ps
      Eq.trans
        (congrArg (fun x => x :: ps.map (evalIncremental evalPair cache)) hd)
        (congrArg (List.cons (evalPair p)) tl)

/-- Theorem 5: Bitset-continuous zero-preserving gate support isomorphism. -/
theorem zero_preserving_gate_support (w pi : Nat) (hw : 0 < w) :
    (0 < w * pi) ↔ (0 < pi) :=
  Iff.intro
    (fun hmul =>
      match pi with
      | 0 => False.elim (Nat.lt_irrefl 0 (Eq.subst (motive := fun z => 0 < z) (Nat.mul_zero w) hmul))
      | n + 1 => Nat.zero_lt_succ n)
    (fun hpi => Nat.mul_pos hw hpi)

-- ============================================================================
-- Section 2. Constructive Zero-Axiom 2x2 Factorial & Shapley Decomposition
-- ============================================================================
-- Represent signed differences without Quotient/Quot.sound via formal pairs (pos - neg).
structure SDiff where
  pos : Nat
  neg : Nat

def sdiffEq (x y : SDiff) : Prop :=
  x.pos + y.neg = y.pos + x.neg

def sdiffAdd (x y : SDiff) : SDiff :=
  ⟨x.pos + y.pos, x.neg + y.neg⟩

def sdiffSub (x y : SDiff) : SDiff :=
  ⟨x.pos + y.neg, x.neg + y.pos⟩

def twiceShapleyEnv (y11 y01 y10 y00 : Nat) : SDiff :=
  sdiffAdd ⟨y11, y01⟩ ⟨y10, y00⟩

def twiceShapleyMat (y11 y01 y10 y00 : Nat) : SDiff :=
  sdiffAdd ⟨y11, y10⟩ ⟨y01, y00⟩

def twiceJointEffect (y11 y00 : Nat) : SDiff :=
  sdiffAdd ⟨y11, y00⟩ ⟨y11, y00⟩

def factorialInteraction (y11 y01 y10 y00 : Nat) : SDiff :=
  ⟨y11 + y00, y01 + y10⟩

/-- Theorem 6 (Exact 2x2 Shapley Efficiency Identity, Zero Axioms):
    `2 * phi_E + 2 * phi_M` equals `2 * (y11 - y00)` for ALL `y11, y01, y10, y00`
    (regardless of synergy or redundancy). -/
theorem shapley_sum_eq_twice_joint (y11 y01 y10 y00 : Nat) :
    sdiffEq
      (sdiffAdd (twiceShapleyEnv y11 y01 y10 y00) (twiceShapleyMat y11 y01 y10 y00))
      (twiceJointEffect y11 y00) :=
  -- LHS: ((y11 + y10) + (y11 + y01)) + (y00 + y00)
  -- RHS: (y11 + y11) + ((y01 + y00) + (y10 + y00))
  have h1 : (y11 + y10) + (y11 + y01) = (y11 + y11) + (y10 + y01) :=
    Eq.trans (Nat.add_assoc y11 y10 (y11 + y01))
      (Eq.trans
        (congrArg (Nat.add y11)
          (Eq.trans (Eq.symm (Nat.add_assoc y10 y11 y01))
            (Eq.trans (congrArg (fun z => z + y01) (Nat.add_comm y10 y11))
              (Nat.add_assoc y11 y10 y01))))
        (Eq.symm (Nat.add_assoc y11 y11 (y10 + y01))))
  have h2 : (y01 + y00) + (y10 + y00) = (y10 + y01) + (y00 + y00) :=
    Eq.trans (Nat.add_assoc y01 y00 (y10 + y00))
      (Eq.trans
        (congrArg (Nat.add y01)
          (Eq.trans (Eq.symm (Nat.add_assoc y00 y10 y00))
            (Eq.trans (congrArg (fun z => z + y00) (Nat.add_comm y00 y10))
              (Nat.add_assoc y10 y00 y00))))
        (Eq.trans (Eq.symm (Nat.add_assoc y01 y10 (y00 + y00)))
          (congrArg (fun z => z + (y00 + y00)) (Nat.add_comm y01 y10))))
  Eq.trans
    (congrArg (fun z => z + (y00 + y00)) h1)
    (Eq.trans
      (Nat.add_assoc (y11 + y11) (y10 + y01) (y00 + y00))
      (congrArg (Nat.add (y11 + y11)) (Eq.symm h2)))

/-- Theorem 7 (Marginal Additivity Gap Equals 2x2 Interaction, Zero Axioms):
    `((y11 - y01) + (y11 - y10)) - (y11 - y00)` equals `y11 - y01 - y10 + y00`. -/
theorem marginal_sum_minus_joint_eq_interaction (y11 y01 y10 y00 : Nat) :
    sdiffEq
      (sdiffSub (sdiffAdd ⟨y11, y01⟩ ⟨y11, y10⟩) ⟨y11, y00⟩)
      (factorialInteraction y11 y01 y10 y00) :=
  -- LHS: ((y11 + y11) + y00) + (y01 + y10)
  -- RHS: (y11 + y00) + ((y01 + y10) + y11)
  have hL : ((y11 + y11) + y00) + (y01 + y10) = (y11 + y00) + (y11 + (y01 + y10)) :=
    Eq.trans
      (congrArg (fun z => z + (y01 + y10))
        (Eq.trans (Nat.add_assoc y11 y11 y00)
          (Eq.trans (congrArg (Nat.add y11) (Nat.add_comm y11 y00))
            (Eq.symm (Nat.add_assoc y11 y00 y11)))))
      (Nat.add_assoc (y11 + y00) y11 (y01 + y10))
  Eq.trans hL (congrArg (Nat.add (y11 + y00)) (Nat.add_comm y11 (y01 + y10)))

-- ============================================================================
-- Section 3. Disjoint-Collision Commutation & F_p = 4 Breakdown Counterexamples
-- ============================================================================

/-- Theorem 8 (Disjoint-Interval Theorem for Modulo-Fp Provenance Routing, Zero Axioms):
    If `p1 = fp * q1 + r` and `p2 = fp * q2 + r` collide in the same modulo-`fp` congruence
    fiber with `p1 < p2` and rule width `ell <= fp`, then `p1 + ell <= p2`, so redex windows
    `[p1, p1 + ell)` and `[p2, p2 + ell2)` are strictly disjoint. -/
theorem mod_collision_disjoint_intervals
    (p1 p2 q1 q2 r fp ell : Nat)
    (hp1 : fp * q1 + r = p1) (hp2 : fp * q2 + r = p2)
    (hlt : p1 < p2) (hell : ell ≤ fp) :
    p1 + ell ≤ p2 :=
  have hq1_lt_q2 : q1 < q2 :=
    match Nat.lt_or_ge q1 q2 with
    | Or.inl h => h
    | Or.inr hge =>
        have hmul : fp * q2 ≤ fp * q1 := Nat.mul_le_mul_left fp hge
        have hsum : fp * q2 + r ≤ fp * q1 + r := Nat.add_le_add_right hmul r
        have hp2_le_p1 : p2 ≤ p1 :=
          Eq.subst (motive := fun x => x ≤ p1) hp2
            (Eq.subst (motive := fun y => fp * q2 + r ≤ y) hp1 hsum)
        False.elim (Nat.not_le_of_gt hlt hp2_le_p1)
  have hq_succ : q1 + 1 ≤ q2 := hq1_lt_q2
  have hmul_succ : fp * (q1 + 1) ≤ fp * q2 := Nat.mul_le_mul_left fp hq_succ
  have hmul_exp : fp * q1 + fp ≤ fp * q2 :=
    Eq.subst (motive := fun z => z ≤ fp * q2)
      (Nat.mul_add fp q1 1 |> Eq.trans <| congrArg (Nat.add (fp * q1)) (Nat.mul_one fp))
      hmul_succ
  have hfp_step : (fp * q1 + r) + fp ≤ fp * q2 + r :=
    have t1 : (fp * q1 + fp) + r ≤ fp * q2 + r := Nat.add_le_add_right hmul_exp r
    have t2 : (fp * q1 + r) + fp = (fp * q1 + fp) + r :=
      Eq.trans (Nat.add_assoc (fp * q1) r fp)
        (Eq.trans (congrArg (Nat.add (fp * q1)) (Nat.add_comm r fp))
          (Eq.symm (Nat.add_assoc (fp * q1) fp r)))
    Eq.subst (motive := fun z => z ≤ fp * q2 + r) (Eq.symm t2) t1
  have hp1_fp : p1 + fp ≤ p2 :=
    Eq.subst (motive := fun y => p1 + fp ≤ y) hp2
      (Eq.subst (motive := fun x => x + fp ≤ fp * q2 + r) hp1 hfp_step)
  Nat.le_trans (Nat.add_le_add_left hell p1) hp1_fp

-- Executable concrete string-rewrite engine & prov_{Fp} x adj2 evaluator
structure SubRule where
  lhs : List Sym
  rhs : List Sym
  deriving DecidableEq

def orientRules : List SubRule → List SubRule
  | [] => []
  | r :: rs => r :: ⟨r.rhs, r.lhs⟩ :: orientRules rs

def listSliceEq : List Sym → List Sym → Bool
  | _, [] => true
  | [], _ :: _ => false
  | x :: xs, y :: ys => (x == y) && listSliceEq xs ys

def replaceAt : Nat → Nat → List Sym → List Sym → List Sym
  | 0, len, rhs, xs => rhs ++ xs.drop len
  | _ + 1, len, rhs, [] => rhs ++ [].drop len
  | p + 1, len, rhs, x :: xs => x :: replaceAt p len rhs xs

def applyRuleAt (r : SubRule) (p : Nat) (w : WordL) : Option WordL :=
  if p + r.lhs.length ≤ w.length && listSliceEq (w.drop p) r.lhs then
    some (replaceAt p r.lhs.length r.rhs w)
  else
    none

def wComm0 : WordL := [.a, .b, .e, .f, .b, .c]
def rComm1 : SubRule := ⟨[.a, .b], [.d, .a]⟩
def rComm2 : SubRule := ⟨[.b, .c], [.f, .a]⟩

/-- Theorem 9: Constructive two-step commutation of mod-4-colliding non-overlapping redexes. -/
theorem disjoint_mod4_rewrites_commute_in_2_steps :
    (match applyRuleAt rComm1 0 wComm0 with
     | some w1 => applyRuleAt rComm2 4 w1
     | none => none) =
    (match applyRuleAt rComm2 4 wComm0 with
     | some w2 => applyRuleAt rComm1 0 w2
     | none => none) ∧
    (match applyRuleAt rComm1 0 wComm0 with
     | some w1 => applyRuleAt rComm2 4 w1
     | none => none) = some [.d, .a, .e, .f, .f, .a] :=
  ⟨rfl, rfl⟩

def dedupWords : List WordL → List WordL
  | [] => []
  | w :: ws => if ws.elem w then dedupWords ws else w :: dedupWords ws

def neighborsWord (ors : List SubRule) (w : WordL) : List (Nat × SubRule × WordL) :=
  let rec loopRules : List SubRule → List (Nat × SubRule × WordL)
    | [] => []
    | r :: rs =>
        let maxP := w.length + 1 - r.lhs.length
        let rec loopPos : Nat → List (Nat × SubRule × WordL)
          | 0 => []
          | k + 1 =>
              let p := k
              match applyRuleAt r p w with
              | some w' =>
                  if w' == w then loopPos k else (p, r, w') :: loopPos k
              | none => loopPos k
        loopPos maxP ++ loopRules rs
  loopRules ors

def concreteStepWords (ors : List SubRule) (S : List WordL) : List WordL :=
  dedupWords (S ++ (S.foldr (fun w acc => (neighborsWord ors w).map (fun x => x.2.2) ++ acc) []))

def concreteIterWords (ors : List SubRule) : Nat → List WordL → List WordL
  | 0, S => dedupWords S
  | k + 1, S => concreteStepWords ors (concreteIterWords ors k S)

abbrev BigramArc := Nat × Sym × Sym

def dedupArcs : List BigramArc → List BigramArc
  | [] => []
  | a :: as => if as.elem a then dedupArcs as else a :: dedupArcs as

def wordToAdj2 (w : WordL) : List BigramArc :=
  let rec go : Nat → List Sym → List BigramArc
    | _, [] => []
    | _, [_] => []
    | p, x :: y :: rest => (p, x, y) :: go (p + 1) (y :: rest)
  go 0 w

def allSyms : List Sym := [.a, .b, .c, .d, .e, .f]

def hasArc (arcs : List BigramArc) (p : Nat) (x y : Sym) : Bool :=
  arcs.elem (p, x, y)

def fwdReach (lm1 : Nat) (arcs : List BigramArc) : Nat → Sym → Bool
  | 0, s => allSyms.any (fun y => hasArc arcs 0 s y)
  | p + 1, s => allSyms.any (fun prev => fwdReach lm1 arcs p prev && hasArc arcs p prev s)

def bwdFrom (lm1 : Nat) (arcs : List BigramArc) : Nat → Sym → Bool
  | 0, s =>
      match lm1 with
      | 0 => false
      | last + 1 => allSyms.any (fun prev => hasArc arcs last prev s)
  | 1, s =>
      allSyms.any (fun nxt => hasArc arcs (lm1 - 1) s nxt)
  | k + 2, s =>
      let q := lm1 - (k + 2)
      allSyms.any (fun nxt => hasArc arcs q s nxt && bwdFrom lm1 arcs (k + 1) nxt)

def trellisFilterAdj2 (lm1 : Nat) (arcs : List BigramArc) : List BigramArc :=
  arcs.filter (fun (p, x, y) => fwdReach lm1 arcs p x && bwdFrom lm1 arcs (lm1 - (p + 1)) y)

def wordInGamma2 (lm1 : Nat) (arcs : List BigramArc) (w : WordL) : Bool :=
  let tf := trellisFilterAdj2 lm1 arcs
  (wordToAdj2 w).all (fun a => tf.elem a)

def adj2StepRuleAt (lm1 : Nat) (ta : List BigramArc) (r : SubRule) (p : Nat) : List BigramArc :=
  match r.lhs, r.rhs with
  | [x0, x1], [y0, y1] =>
      if p < lm1 && hasArc ta p x0 x1 then
        let core := [(p, y0, y1)]
        let leftB :=
          if p > 0 then
            (allSyms.filter (fun s => hasArc ta (p - 1) s x0)).map (fun s => (p - 1, s, y0))
          else []
        let rightB :=
          if p + 1 < lm1 then
            (allSyms.filter (fun s => hasArc ta (p + 1) x1 s)).map (fun s => (p + 1, y1, s))
          else []
        core ++ leftB ++ rightB
      else []
  | [x0, x1, x2], [y0, y1, y2] =>
      if p + 1 < lm1 && hasArc ta p x0 x1 && hasArc ta (p + 1) x1 x2 then
        let core := [(p, y0, y1), (p + 1, y1, y2)]
        let leftB :=
          if p > 0 then
            (allSyms.filter (fun s => hasArc ta (p - 1) s x0)).map (fun s => (p - 1, s, y0))
          else []
        let rightB :=
          if p + 2 < lm1 then
            (allSyms.filter (fun s => hasArc ta (p + 2) x2 s)).map (fun s => (p + 2, y2, s))
          else []
        core ++ leftB ++ rightB
      else []
  | _, _ => []

def adj2TransferStep (lm1 : Nat) (ors : List SubRule) (w0 : WordL)
    (fp : Nat) (fIdx : Nat) (suppressRootBacktrack : Bool) (arcs : List BigramArc) : List BigramArc :=
  let ta := trellisFilterAdj2 lm1 arcs
  let positions := List.range lm1
  let added := ors.foldr (fun r acc =>
    (positions.foldr (fun p acc2 =>
      let skip := suppressRootBacktrack && (p % fp == fIdx) &&
        (p + r.rhs.length ≤ w0.length && listSliceEq (w0.drop p) r.rhs)
      if skip then acc2 else adj2StepRuleAt lm1 ta r p ++ acc2) []) ++ acc) []
  trellisFilterAdj2 lm1 (dedupArcs (ta ++ added))

def runProvFpAdj2 (ors : List SubRule) (w0 : WordL) (fp : Nat) (kMax : Nat) : List (List BigramArc) :=
  let lm1 := w0.length - 1
  let base := wordToAdj2 w0
  match kMax with
  | 0 => [base]
  | k + 1 =>
      let step1 := neighborsWord ors w0
      let initFibs := (List.range fp).map (fun f =>
        let matching := step1.filter (fun (p, _, _) => p % fp == f)
        let raw := matching.foldr (fun (_, _, w1) acc => wordToAdj2 w1 ++ acc) []
        if matching.isEmpty then [] else trellisFilterAdj2 lm1 (dedupArcs raw))
      let rec loopSteps : Nat → Nat → List (List BigramArc) → List (List BigramArc)
        | 0, _, fibs => fibs
        | m + 1, stepIdx, fibs =>
            let nextFibs := (List.range fp).zip fibs |>.map (fun (f, fa) =>
              if fa.isEmpty then []
              else adj2TransferStep lm1 ors w0 fp f (stepIdx == 1) fa)
            loopSteps m (stepIdx + 1) nextFibs
      base :: loopSteps k 1 initFibs

def wordAcceptedByProvFpAdj2 (ors : List SubRule) (w0 : WordL) (fp : Nat) (kMax : Nat) (cand : WordL) : Bool :=
  let lm1 := w0.length - 1
  let fibs := runProvFpAdj2 ors w0 fp kMax
  fibs.any (fun fa => !fa.isEmpty && wordInGamma2 lm1 fa cand)

-- Counterexample 4A: Rule width l = 5 > Fp = 4 ("abbbebbbc")
def w4a : WordL := [.a, .b, .b, .b, .e, .b, .b, .b, .c]
def chimera4a : WordL := [.d, .b, .b, .b, .f, .b, .b, .b, .d]
def rules4a : List SubRule := orientRules [
  ⟨[.a, .b, .b, .b, .e], [.d, .b, .b, .b, .f]⟩,
  ⟨[.e, .b, .b, .b, .c], [.f, .b, .b, .b, .d]⟩
]

theorem cex4a_fp4_spurious_fp5_exact :
    (concreteIterWords rules4a 3 [w4a]).length = 3 ∧
    (concreteIterWords rules4a 3 [w4a]).elem chimera4a = false ∧
    wordAcceptedByProvFpAdj2 rules4a w4a 4 1 chimera4a = true ∧
    wordAcceptedByProvFpAdj2 rules4a w4a 5 1 chimera4a = false :=
  ⟨rfl, ⟨rfl, ⟨rfl, rfl⟩⟩⟩

-- Counterexample 4B: Secondary Step-2 Intra-Fiber Forking (l <= 3, Fp = 4 & Fp = 8)
def w4b : WordL := [.f, .a, .b, .b, .c]
def rules4b : List SubRule := orientRules [
  ⟨[.f, .a], [.a, .a]⟩,
  ⟨[.a, .b], [.d, .e]⟩,
  ⟨[.b, .b, .c], [.e, .f, .a]⟩,
  ⟨[.d, .e, .f], [.c, .c, .c]⟩
]

def hasSymbolAtPos1 (words : List WordL) (s : Sym) : Bool :=
  words.any (fun w => match w with | _ :: x :: _ => x == s | _ => false)

def provFpHasSymbolAtPos1 (ors : List SubRule) (w0 : WordL) (fp kMax : Nat) (s : Sym) : Bool :=
  let fibs := runProvFpAdj2 ors w0 fp kMax
  fibs.any (fun fa => fa.any (fun (p, x, _) => p == 1 && x == s))

theorem cex4b_intrafiber_step2_fork :
    hasSymbolAtPos1 (concreteIterWords rules4b 3 [w4b]) .c = false ∧
    provFpHasSymbolAtPos1 rules4b w4b 4 3 .c = true ∧
    provFpHasSymbolAtPos1 rules4b w4b 8 3 .c = true :=
  ⟨rfl, ⟨rfl, rfl⟩⟩

-- Counterexample 4C: Horizon Compression on Longer Words (L = 10, k = 2, Fp = 4 vs Fp = 10)
def w4c : WordL := [.a, .b, .e, .f, .a, .b, .e, .f, .a, .b]
def target4c : WordL := [.c, .d, .e, .f, .c, .d, .e, .f, .c, .d]
def rules4c : List SubRule := orientRules [
  ⟨[.a, .b], [.c, .d]⟩
]

theorem cex4c_horizon_compression_long_word :
    (concreteIterWords rules4c 2 [w4c]).elem target4c = false ∧
    (concreteIterWords rules4c 3 [w4c]).elem target4c = true ∧
    wordAcceptedByProvFpAdj2 rules4c w4c 4 1 target4c = true ∧
    wordAcceptedByProvFpAdj2 rules4c w4c 10 1 target4c = false :=
  ⟨rfl, ⟨rfl, ⟨rfl, rfl⟩⟩⟩

-- ============================================================================
-- Section 4. Theorem A: General Soundness of Guarded Congruence-Fibered Adj2
--            (Fable Modulus/Length Guard & PR #348 Changed-Coordinate Guard)
-- ============================================================================

def WordF (N : Nat) : Type := Fin N → Sym

def WordEq {N : Nat} (u v : WordF N) : Prop :=
  ∀ j : Fin N, u j = v j

theorem wordEq_refl {N : Nat} (u : WordF N) : WordEq u u :=
  fun _ => rfl

theorem wordEq_symm {N : Nat} {u v : WordF N} (h : WordEq u v) : WordEq v u :=
  fun j => Eq.symm (h j)

theorem wordEq_trans {N : Nat} {u v w : WordF N} (h1 : WordEq u v) (h2 : WordEq v w) : WordEq u w :=
  fun j => Eq.trans (h1 j) (h2 j)

-- Constructive zero-axiom quotient & remainder for modulus m = mPred + 1
def divModSucc (mPred : Nat) : Nat → Nat × Nat
  | 0 => (0, 0)
  | p + 1 =>
      let qr := divModSucc mPred p
      if qr.2 = mPred then (qr.1 + 1, 0) else (qr.1, qr.2 + 1)

def quotPos (mPred p : Nat) : Nat := (divModSucc mPred p).1
def modPos (mPred p : Nat) : Nat := (divModSucc mPred p).2

theorem divModSucc_spec (mPred : Nat) :
    ∀ p : Nat, (mPred + 1) * quotPos mPred p + modPos mPred p = p
  | 0 => Nat.mul_zero (mPred + 1)
  | p + 1 =>
      let qr := divModSucc mPred p
      have ih : (mPred + 1) * qr.1 + qr.2 = p := divModSucc_spec mPred p
      match decEq qr.2 mPred with
      | isTrue h_eq =>
          have h_if : (if qr.2 = mPred then (qr.1 + 1, 0) else (qr.1, qr.2 + 1)) = (qr.1 + 1, 0) :=
            if_pos h_eq
          have hq : quotPos mPred (p + 1) = qr.1 + 1 := congrArg Prod.fst h_if
          have hr : modPos mPred (p + 1) = 0 := congrArg Prod.snd h_if
          have h_calc : (mPred + 1) * (qr.1 + 1) + 0 = p + 1 :=
            Eq.trans (Nat.add_zero ((mPred + 1) * (qr.1 + 1)))
              (Eq.trans (Nat.mul_add (mPred + 1) qr.1 1)
                (Eq.trans (congrArg (Nat.add ((mPred + 1) * qr.1)) (Nat.mul_one (mPred + 1)))
                  (Eq.trans (congrArg (fun z => (mPred + 1) * qr.1 + (z + 1)) (Eq.symm h_eq))
                    (Eq.trans (Eq.symm (Nat.add_assoc ((mPred + 1) * qr.1) qr.2 1))
                      (congrArg (fun z => z + 1) ih)))))
          Eq.trans
            (congrArg (fun z => z + modPos mPred (p + 1)) (congrArg (Nat.mul (mPred + 1)) hq))
            (Eq.trans (congrArg (Nat.add ((mPred + 1) * (qr.1 + 1))) hr) h_calc)
      | isFalse h_ne =>
          have h_def : divModSucc mPred (p + 1) =
              (if qr.2 = mPred then (qr.1 + 1, 0) else (qr.1, qr.2 + 1)) := rfl
          have h_if : (if qr.2 = mPred then (qr.1 + 1, 0) else (qr.1, qr.2 + 1)) = (qr.1, qr.2 + 1) :=
            if_neg h_ne
          have h_pair : divModSucc mPred (p + 1) = (qr.1, qr.2 + 1) := Eq.trans h_def h_if
          have hq : quotPos mPred (p + 1) = qr.1 :=
            @congrArg (Nat × Nat) Nat (divModSucc mPred (p + 1)) (qr.1, qr.2 + 1) Prod.fst h_pair
          have hr : modPos mPred (p + 1) = qr.2 + 1 :=
            @congrArg (Nat × Nat) Nat (divModSucc mPred (p + 1)) (qr.1, qr.2 + 1) Prod.snd h_pair
          have h_calc : (mPred + 1) * qr.1 + (qr.2 + 1) = p + 1 :=
            Eq.trans (Eq.symm (Nat.add_assoc ((mPred + 1) * qr.1) qr.2 1))
              (congrArg (fun z => z + 1) ih)
          Eq.trans
            (congrArg (fun z => z + modPos mPred (p + 1)) (congrArg (Nat.mul (mPred + 1)) hq))
            (Eq.trans (congrArg (Nat.add ((mPred + 1) * qr.1)) hr) h_calc)

theorem modPos_le (mPred : Nat) : ∀ p : Nat, modPos mPred p ≤ mPred
  | 0 => Nat.zero_le mPred
  | p + 1 =>
      let qr := divModSucc mPred p
      have ih : qr.2 ≤ mPred := modPos_le mPred p
      match decEq qr.2 mPred with
      | isTrue h_eq =>
          have hr : modPos mPred (p + 1) = 0 := congrArg Prod.snd (if_pos h_eq)
          Eq.subst (motive := fun z => z ≤ mPred) (Eq.symm hr) (Nat.zero_le mPred)
      | isFalse h_ne =>
          have hr : modPos mPred (p + 1) = qr.2 + 1 := congrArg Prod.snd (if_neg h_ne)
          have hlt : qr.2 < mPred :=
            match Nat.lt_or_ge qr.2 mPred with
            | Or.inl h => h
            | Or.inr hge => False.elim (h_ne (Nat.le_antisymm ih hge))
          Eq.subst (motive := fun z => z ≤ mPred) (Eq.symm hr) hlt

theorem modPos_disjoint_intervals (mPred p1 p2 ell : Nat)
    (hmod : modPos mPred p1 = modPos mPred p2)
    (hlt : p1 < p2) (hell : ell ≤ mPred + 1) :
    p1 + ell ≤ p2 :=
  have hp1 : (mPred + 1) * quotPos mPred p1 + modPos mPred p1 = p1 := divModSucc_spec mPred p1
  have hp2_raw : (mPred + 1) * quotPos mPred p2 + modPos mPred p2 = p2 := divModSucc_spec mPred p2
  have hp2 : (mPred + 1) * quotPos mPred p2 + modPos mPred p1 = p2 :=
    Eq.subst (motive := fun r => (mPred + 1) * quotPos mPred p2 + r = p2) (Eq.symm hmod) hp2_raw
  mod_collision_disjoint_intervals p1 p2 (quotPos mPred p1) (quotPos mPred p2)
    (modPos mPred p1) (mPred + 1) ell hp1 hp2 hlt hell

-- Position-indexed adjacent-pair domain Abs2 on words of length L + 1
def Abs2 (L : Nat) : Type := Fin L → Sym → Sym → Prop

def leAbs2 {L : Nat} (A B : Abs2 L) : Prop :=
  ∀ q s0 s1, A q s0 s1 → B q s0 s1

def alpha2 {L : Nat} (S : WordF (L + 1) → Prop) : Abs2 L :=
  fun q s0 s1 => ∃ w, S w ∧ w q.castSucc = s0 ∧ w q.succ = s1

def gamma2 {L : Nat} (A : Abs2 L) : WordF (L + 1) → Prop :=
  fun w => ∀ q : Fin L, A q (w q.castSucc) (w q.succ)

def trellisReduce2 {L : Nat} (A : Abs2 L) : Abs2 L :=
  alpha2 (gamma2 A)

theorem gamma2_wordEq {L : Nat} {A : Abs2 L} {u v : WordF (L + 1)}
    (heq : WordEq u v) (hu : gamma2 A u) : gamma2 A v :=
  fun q =>
    Eq.subst (motive := fun x => A q x (v q.succ)) (heq q.castSucc)
      (Eq.subst (motive := fun y => A q (u q.castSucc) y) (heq q.succ) (hu q))

theorem gamma2_mono {L : Nat} {A B : Abs2 L} (hAB : leAbs2 A B)
    {w : WordF (L + 1)} (hw : gamma2 A w) : gamma2 B w :=
  fun q => hAB q (w q.castSucc) (w q.succ) (hw q)

theorem gamma2_trellisReduce2 {L : Nat} {A : Abs2 L} {w : WordF (L + 1)}
    (hw : gamma2 A w) : gamma2 (trellisReduce2 A) w :=
  fun _ => ⟨w, hw, rfl, rfl⟩

private theorem fin_eq_of_val_eq {n : Nat} : ∀ {i j : Fin n}, i.val = j.val → i = j
  | ⟨_, _⟩, ⟨_, _⟩, rfl => rfl

/-- Singleton exactness of Abs2 for words of length >= 2 (L >= 1):
    `gamma2 (alpha2 (WordEq w0)) w` implies `WordEq w0 w`. -/
theorem gamma2_singleton_exact {L : Nat} (hL : 0 < L) (w0 w : WordF (L + 1))
    (hw : gamma2 (alpha2 (WordEq w0)) w) : WordEq w0 w :=
  fun j =>
    match L, hL, w0, w, hw, j with
    | m + 1, _, _, w, hw, ⟨0, h0⟩ =>
        let q : Fin (m + 1) := ⟨0, Nat.zero_lt_succ m⟩
        have hq : q.castSucc = ⟨0, h0⟩ := fin_eq_of_val_eq rfl
        match hw q with
        | ⟨u, hu0, hu_cs, _⟩ =>
            Eq.trans (hu0 ⟨0, h0⟩)
              (Eq.trans (congrArg u (Eq.symm hq))
                (Eq.trans hu_cs (congrArg w hq)))
    | m + 1, _, _, w, hw, ⟨k + 1, hk⟩ =>
        let q : Fin (m + 1) := ⟨k, Nat.lt_of_succ_lt_succ hk⟩
        have hq : q.succ = ⟨k + 1, hk⟩ := fin_eq_of_val_eq rfl
        match hw q with
        | ⟨u, hu0, _, hu_s⟩ =>
            Eq.trans (hu0 ⟨k + 1, hk⟩)
              (Eq.trans (congrArg u (Eq.symm hq))
                (Eq.trans hu_s (congrArg w hq)))

-- Contiguous window rewrites on WordF (L + 1)
structure WindowRule (N : Nat) where
  pos   : Nat
  len   : Nat
  lhs   : Fin N → Sym
  rhs   : Fin N → Sym
  hlen2 : 2 ≤ len
  hbnd  : pos + len ≤ N

def inWin (p len j : Nat) : Bool :=
  decide (p ≤ j) && decide (j < p + len)

theorem inWin_true_iff (p len j : Nat) :
    inWin p len j = true ↔ (p ≤ j ∧ j < p + len) :=
  Iff.intro
    (fun h =>
      match decLe : decide (p ≤ j), decLt : decide (j < p + len) with
      | true, true => ⟨of_decide_eq_true decLe, of_decide_eq_true decLt⟩
      | false, _ =>
          False.elim (Bool.noConfusion (Eq.trans (Eq.symm (congrArg (fun b => b && decide (j < p + len)) decLe)) h))
      | true, false =>
          False.elim (Bool.noConfusion (Eq.trans (Eq.symm (Eq.trans (congrArg (fun b => b && decide (j < p + len)) decLe) (congrArg (and true) decLt))) h)))
    (fun ⟨hle, hlt⟩ =>
      Eq.trans (congrArg (fun b => b && decide (j < p + len)) (decide_eq_true hle))
        (decide_eq_true hlt))

def MatchesWin {N : Nat} (r : WindowRule N) (w : WordF N) : Prop :=
  ∀ j : Fin N, inWin r.pos r.len j.val = true → w j = r.lhs j

def applyWin {N : Nat} (r : WindowRule N) (w : WordF N) : WordF N :=
  fun j => if inWin r.pos r.len j.val then r.rhs j else w j

theorem matchesWin_wordEq {N : Nat} {r : WindowRule N} {u v : WordF N}
    (heq : WordEq u v) (hu : MatchesWin r u) : MatchesWin r v :=
  fun j hj => Eq.trans (Eq.symm (heq j)) (hu j hj)

theorem applyWin_wordEq {N : Nat} (r : WindowRule N) {u v : WordF N}
    (heq : WordEq u v) : WordEq (applyWin r u) (applyWin r v) :=
  fun j =>
    match h : inWin r.pos r.len j.val with
    | true =>
        Eq.trans (if_pos h) (Eq.symm (if_pos h))
    | false =>
        have hnot : ¬(inWin r.pos r.len j.val = true) := fun ht => Bool.noConfusion (Eq.trans (Eq.symm h) ht)
        Eq.trans (if_neg hnot) (Eq.trans (heq j) (Eq.symm (if_neg hnot)))

def ConcreteStep {N : Nat} (R : List (WindowRule N)) (S : WordF N → Prop) : WordF N → Prop :=
  fun w' => S w' ∨ ∃ w, S w ∧ ∃ r, r ∈ R ∧ MatchesWin r w ∧ WordEq (applyWin r w) w'

def IterConcreteWin {N : Nat} (R : List (WindowRule N)) : Nat → (WordF N → Prop) → (WordF N → Prop)
  | 0, S => S
  | k + 1, S => ConcreteStep R (IterConcreteWin R k S)

-- Executable universal/existential quantifiers over Fin N (zero axioms)
def finAll : (N : Nat) → (Fin N → Bool) → Bool
  | 0, _ => true
  | n + 1, pred => pred ⟨0, Nat.zero_lt_succ n⟩ && finAll n (fun i => pred i.succ)

private theorem and_left_true : ∀ {x y : Bool}, (x && y) = true → x = true
  | true, _, _ => rfl
  | false, _, h => Bool.noConfusion h

private theorem and_right_true : ∀ {x y : Bool}, (x && y) = true → y = true
  | true, true, _ => rfl
  | false, _, h => Bool.noConfusion h
  | true, false, h => Bool.noConfusion h

theorem finAll_true_forall : ∀ {N : Nat} {pred : Fin N → Bool},
    finAll N pred = true → ∀ j : Fin N, pred j = true
  | 0, _, _, ⟨_, hj⟩ => False.elim (Nat.not_lt_zero _ hj)
  | n + 1, pred, hall, ⟨0, h0⟩ =>
      have h_eq : (⟨0, Nat.zero_lt_succ n⟩ : Fin (n + 1)) = ⟨0, h0⟩ := fin_eq_of_val_eq rfl
      have h_hd : pred ⟨0, Nat.zero_lt_succ n⟩ = true := and_left_true hall
      Eq.subst (motive := fun x => pred x = true) h_eq h_hd
  | n + 1, pred, hall, ⟨k + 1, hk⟩ =>
      let q : Fin n := ⟨k, Nat.lt_of_succ_lt_succ hk⟩
      have h_eq : q.succ = ⟨k + 1, hk⟩ := fin_eq_of_val_eq rfl
      have h_tl : finAll n (fun i => pred i.succ) = true := and_right_true hall
      Eq.subst (motive := fun x => pred x = true) h_eq (finAll_true_forall h_tl q)

def matchesWinBool {N : Nat} (r : WindowRule N) (w : WordF N) : Bool :=
  finAll N (fun j => if inWin r.pos r.len j.val then decide (w j = r.lhs j) else true)

theorem matchesWinBool_of_matchesWin {N : Nat} {r : WindowRule N} {w : WordF N}
    (hm : MatchesWin r w) : matchesWinBool r w = true :=
  let rec go : ∀ (M : Nat) (f : Fin M → Fin N),
      finAll M (fun i => if inWin r.pos r.len (f i).val then decide (w (f i) = r.lhs (f i)) else true) = true
    | 0, _ => rfl
    | m + 1, f =>
        have h0 : (if inWin r.pos r.len (f ⟨0, Nat.zero_lt_succ m⟩).val
                   then decide (w (f ⟨0, Nat.zero_lt_succ m⟩) = r.lhs (f ⟨0, Nat.zero_lt_succ m⟩))
                   else true) = true :=
          match hw : inWin r.pos r.len (f ⟨0, Nat.zero_lt_succ m⟩).val with
          | true => decide_eq_true (hm (f ⟨0, Nat.zero_lt_succ m⟩) hw)
          | false => rfl
        have hrest := go m (fun i => f i.succ)
        Eq.trans (congrArg (fun b => b && _) h0) hrest
  go N (fun j => j)

def rhsAgreesBool {N : Nat} (w0 : WordF N) (r : WindowRule N) : Bool :=
  finAll N (fun j => if inWin r.pos r.len j.val then decide (r.rhs j = w0 j) else true)

theorem rhsAgrees_of_bool {N : Nat} {w0 : WordF N} {r : WindowRule N}
    (hb : rhsAgreesBool w0 r = true) :
    ∀ j : Fin N, inWin r.pos r.len j.val = true → r.rhs j = w0 j :=
  fun j hj =>
    have hall := finAll_true_forall hb j
    have hdec : decide (r.rhs j = w0 j) = true := Eq.trans (Eq.symm (if_pos hj)) hall
    of_decide_eq_true hdec

theorem listAll_mem_true {α : Type} {p : α → Bool} :
    ∀ {xs : List α}, xs.all p = true → ∀ {x : α}, x ∈ xs → p x = true
  | [], _, _, hmem => nomatch hmem
  | _ :: _, hall, _, List.Mem.head _ => and_left_true hall
  | _ :: _, hall, _, List.Mem.tail _ htail => listAll_mem_true (and_right_true hall) htail

private theorem or_intro_left_true : ∀ (b : Bool), (true || b) = true := fun _ => rfl
private theorem or_intro_right_true : ∀ (x : Bool), (x || true) = true
  | true => rfl
  | false => rfl

theorem listAny_of_mem_true {α : Type} {p : α → Bool} :
    ∀ {xs : List α} {x : α}, x ∈ xs → p x = true → xs.any p = true
  | [], _, hmem, _ => nomatch hmem
  | _ :: ys, _, List.Mem.head _, hx =>
      Eq.trans (congrArg (fun b => b || ys.any p) hx) (or_intro_left_true (ys.any p))
  | y :: _, _, List.Mem.tail _ htail, hx =>
      have ih := listAny_of_mem_true htail hx
      Eq.trans (congrArg (fun b => p y || b) ih) (or_intro_right_true (p y))

-- Local adjacent-bigram matching & boundary-stitching transfer (exact mirror of Python/Rust cell)
def AbstractMatchesAdj2 {L : Nat} (r : WindowRule (L + 1)) (T : Abs2 L) : Prop :=
  ∀ q : Fin L,
    inWin r.pos r.len q.val = true →
    inWin r.pos r.len (q.val + 1) = true →
    T q (r.lhs q.castSucc) (r.lhs q.succ)

def LocalAdj2RuleEmit {L : Nat} (r : WindowRule (L + 1)) (T : Abs2 L)
    (q : Fin L) (s0 s1 : Sym) : Prop :=
  AbstractMatchesAdj2 r T ∧
  ((inWin r.pos r.len q.val = true ∧ inWin r.pos r.len (q.val + 1) = true ∧
      s0 = r.rhs q.castSucc ∧ s1 = r.rhs q.succ) ∨
   (inWin r.pos r.len q.val = false ∧ inWin r.pos r.len (q.val + 1) = true ∧
      T q s0 (r.lhs q.succ) ∧ s1 = r.rhs q.succ) ∨
   (inWin r.pos r.len q.val = true ∧ inWin r.pos r.len (q.val + 1) = false ∧
      T q (r.lhs q.castSucc) s1 ∧ s0 = r.rhs q.castSucc))

def LocalAdj2GuardedStep {L : Nat} (R : List (WindowRule (L + 1)))
    (suppress : WindowRule (L + 1) → Bool) (A : Abs2 L) : Abs2 L :=
  let T := trellisReduce2 A
  trellisReduce2 (fun q s0 s1 =>
    T q s0 s1 ∨ ∃ r, r ∈ R ∧ suppress r = false ∧ LocalAdj2RuleEmit r T q s0 s1)

theorem localAdj2GuardedStep_refl {L : Nat} (R : List (WindowRule (L + 1)))
    (suppress : WindowRule (L + 1) → Bool) {A : Abs2 L} {w : WordF (L + 1)}
    (hw : gamma2 A w) : gamma2 (LocalAdj2GuardedStep R suppress A) w :=
  have hT : gamma2 (trellisReduce2 A) w := gamma2_trellisReduce2 hw
  gamma2_trellisReduce2 (fun q => Or.inl (hT q))

/-- Step soundness of an unsuppressed firing under the local adjacent-bigram transfer. -/
theorem localAdj2_unsuppressed_firing_sound {L : Nat}
    (R : List (WindowRule (L + 1))) (suppress : WindowRule (L + 1) → Bool)
    {A : Abs2 L} {w w' : WordF (L + 1)} {r : WindowRule (L + 1)}
    (hw : gamma2 A w) (hr : r ∈ R) (hns : suppress r = false)
    (hm : MatchesWin r w) (heq : WordEq (applyWin r w) w') :
    gamma2 (LocalAdj2GuardedStep R suppress A) w' :=
  let T := trellisReduce2 A
  have hTw : gamma2 T w := gamma2_trellisReduce2 hw
  have hAbsMatch : AbstractMatchesAdj2 r T :=
    fun q hq0 hq1 =>
      have e0 : w q.castSucc = r.lhs q.castSucc := hm q.castSucc hq0
      have e1 : w q.succ = r.lhs q.succ := hm q.succ hq1
      Eq.subst (motive := fun x => T q x (r.lhs q.succ)) e0
        (Eq.subst (motive := fun y => T q (w q.castSucc) y) e1 (hTw q))
  have hApply : gamma2 (LocalAdj2GuardedStep R suppress A) (applyWin r w) :=
    gamma2_trellisReduce2 (fun q =>
      match h0 : inWin r.pos r.len q.val, h1 : inWin r.pos r.len (q.val + 1) with
      | true, true =>
          have v0 : applyWin r w q.castSucc = r.rhs q.castSucc := if_pos h0
          have v1 : applyWin r w q.succ = r.rhs q.succ := if_pos h1
          Or.inr ⟨r, hr, hns, hAbsMatch, Or.inl ⟨h0, h1, v0, v1⟩⟩
      | false, true =>
          have hn0 : ¬(inWin r.pos r.len q.val = true) := fun ht => Bool.noConfusion (Eq.trans (Eq.symm h0) ht)
          have v0 : applyWin r w q.castSucc = w q.castSucc := if_neg hn0
          have v1 : applyWin r w q.succ = r.rhs q.succ := if_pos h1
          have el1 : w q.succ = r.lhs q.succ := hm q.succ h1
          have hTq : T q (applyWin r w q.castSucc) (r.lhs q.succ) :=
            Eq.subst (motive := fun x => T q x (r.lhs q.succ)) (Eq.symm v0)
              (Eq.subst (motive := fun y => T q (w q.castSucc) y) el1 (hTw q))
          Or.inr ⟨r, hr, hns, hAbsMatch, Or.inr (Or.inl ⟨h0, h1, hTq, v1⟩)⟩
      | true, false =>
          have hn1 : ¬(inWin r.pos r.len (q.val + 1) = true) := fun ht => Bool.noConfusion (Eq.trans (Eq.symm h1) ht)
          have v0 : applyWin r w q.castSucc = r.rhs q.castSucc := if_pos h0
          have v1 : applyWin r w q.succ = w q.succ := if_neg hn1
          have el0 : w q.castSucc = r.lhs q.castSucc := hm q.castSucc h0
          have hTq : T q (r.lhs q.castSucc) (applyWin r w q.succ) :=
            Eq.subst (motive := fun y => T q (r.lhs q.castSucc) y) (Eq.symm v1)
              (Eq.subst (motive := fun x => T q x (w q.succ)) el0 (hTw q))
          Or.inr ⟨r, hr, hns, hAbsMatch, Or.inr (Or.inr ⟨h0, h1, hTq, v0⟩)⟩
      | false, false =>
          have hn0 : ¬(inWin r.pos r.len q.val = true) := fun ht => Bool.noConfusion (Eq.trans (Eq.symm h0) ht)
          have hn1 : ¬(inWin r.pos r.len (q.val + 1) = true) := fun ht => Bool.noConfusion (Eq.trans (Eq.symm h1) ht)
          have v0 : applyWin r w q.castSucc = w q.castSucc := if_neg hn0
          have v1 : applyWin r w q.succ = w q.succ := if_neg hn1
          Or.inl
            (Eq.subst (motive := fun x => T q x (applyWin r w q.succ)) (Eq.symm v0)
              (Eq.subst (motive := fun y => T q (w q.castSucc) y) (Eq.symm v1) (hTw q))))
  gamma2_wordEq heq hApply

-- Executable Boolean Guard Predicates:
-- (A) Fable's Modulus/Length Guard (`fableLenSuppressBool`)
def fableLenSuppressBool {N : Nat} (R : List (WindowRule N)) (w0 : WordF N)
    (mPred f : Nat) (r : WindowRule N) : Bool :=
  decide (modPos mPred r.pos = f) &&
  rhsAgreesBool w0 r &&
  R.all (fun r1 =>
    if matchesWinBool r1 w0 && decide (r1.pos = r.pos) then decide (r1.len ≤ r.len) else true)

-- (B) PR #348's Conservative Changed-Coordinate Mask Guard (`pr348MaskSuppressBool`)
def pr348MaskSuppressBool {N : Nat} (R : List (WindowRule N)) (w0 : WordF N)
    (mPred f : Nat) (r : WindowRule N) : Bool :=
  decide (modPos mPred r.pos = f) &&
  rhsAgreesBool w0 r &&
  finAll N (fun j =>
    if R.any (fun r1 =>
      matchesWinBool r1 w0 && decide (modPos mPred r1.pos = f) && decide (applyWin r1 w0 j ≠ w0 j))
    then inWin r.pos r.len j.val
    else true)

private theorem and3_eq_true : ∀ {x y z : Bool}, (x && y && z) = true → x = true ∧ y = true ∧ z = true
  | true, true, true, _ => ⟨rfl, rfl, rfl⟩
  | false, _, _, h => Bool.noConfusion h
  | true, false, _, h => Bool.noConfusion h
  | true, true, false, h => Bool.noConfusion h

/-- Core Lemma A1 (Fable Guard Safety): Under `ellMax <= mPred + 1` (`m >= ell_max`),
    any concrete 2-step sequence `w0 ->^{r1} w1 ->^{r2} w2` in fiber `f = modPos mPred r1.pos`
    whose second step satisfies `fableLenSuppressBool = true` produces EITHER `w0` OR `w1`! -/
theorem fable_suppressed_pass2_returns_root_or_stays {N : Nat}
    (R : List (WindowRule N)) (w0 : WordF N) (mPred ellMax : Nat)
    (hMod : ellMax ≤ mPred + 1)
    (hWidths : ∀ r, r ∈ R → r.len ≤ ellMax)
    {r1 r2 : WindowRule N}
    (hr1 : r1 ∈ R) (hm1 : MatchesWin r1 w0)
    (hr2 : r2 ∈ R) (_hm2 : MatchesWin r2 (applyWin r1 w0))
    (hSup : fableLenSuppressBool R w0 mPred (modPos mPred r1.pos) r2 = true) :
    WordEq w0 (applyWin r2 (applyWin r1 w0)) ∨
    WordEq (applyWin r1 w0) (applyWin r2 (applyWin r1 w0)) :=
  let ⟨hd_mod, hb_rhs, hall_r⟩ := and3_eq_true hSup
  have hmod_eq : modPos mPred r2.pos = modPos mPred r1.pos := of_decide_eq_true hd_mod
  have hrhs2 : ∀ j : Fin N, inWin r2.pos r2.len j.val = true → r2.rhs j = w0 j := rhsAgrees_of_bool hb_rhs
  match decEq r1.pos r2.pos with
  | isTrue hp_eq =>
      have hr1_chk := listAll_mem_true hall_r hr1
      have hm1_b : matchesWinBool r1 w0 = true := matchesWinBool_of_matchesWin hm1
      have hp_b : decide (r1.pos = r2.pos) = true := decide_eq_true hp_eq
      have hcond : (matchesWinBool r1 w0 && decide (r1.pos = r2.pos)) = true :=
        Eq.trans (congrArg (fun x => x && decide (r1.pos = r2.pos)) hm1_b) hp_b
      have hlen_le : r1.len ≤ r2.len :=
        of_decide_eq_true (Eq.trans (Eq.symm (if_pos hcond)) hr1_chk)
      Or.inl (fun j =>
        match h2 : inWin r2.pos r2.len j.val with
        | true => Eq.symm (Eq.trans (if_pos h2) (hrhs2 j h2))
        | false =>
            have hn2 : ¬(inWin r2.pos r2.len j.val = true) :=
              fun ht => Bool.noConfusion (Eq.trans (Eq.symm h2) ht)
            have hn1 : ¬(inWin r1.pos r1.len j.val = true) :=
              fun ht1 =>
                have ⟨hp1_le, hj_lt1⟩ := (inWin_true_iff r1.pos r1.len j.val).mp ht1
                have hp2_le : r2.pos ≤ j.val := Eq.subst (motive := fun p => p ≤ j.val) hp_eq hp1_le
                have hbound12 : r1.pos + r1.len ≤ r2.pos + r2.len :=
                  Eq.subst (motive := fun p => r1.pos + r1.len ≤ p + r2.len) hp_eq
                    (Nat.add_le_add_left hlen_le r1.pos)
                have hj_lt2 : j.val < r2.pos + r2.len := Nat.lt_of_lt_of_le hj_lt1 hbound12
                hn2 ((inWin_true_iff r2.pos r2.len j.val).mpr ⟨hp2_le, hj_lt2⟩)
            Eq.symm (Eq.trans (if_neg hn2) (if_neg hn1)))
  | isFalse hp_ne =>
      have hDisj : ∀ j : Fin N, inWin r2.pos r2.len j.val = true → ¬(inWin r1.pos r1.len j.val = true) :=
        fun j hj2 ht1 =>
          have ⟨hp2_le, hj_lt2⟩ := (inWin_true_iff r2.pos r2.len j.val).mp hj2
          have ⟨hp1_le, hj_lt1⟩ := (inWin_true_iff r1.pos r1.len j.val).mp ht1
          match Nat.lt_or_ge r1.pos r2.pos with
          | Or.inl hlt12 =>
              have hell1 : r1.len ≤ mPred + 1 := Nat.le_trans (hWidths r1 hr1) hMod
              have hdis : r1.pos + r1.len ≤ r2.pos :=
                modPos_disjoint_intervals mPred r1.pos r2.pos r1.len (Eq.symm hmod_eq) hlt12 hell1
              Nat.not_le_of_gt hj_lt1 (Nat.le_trans hdis hp2_le)
          | Or.inr hge12 =>
              have hlt21 : r2.pos < r1.pos :=
                match Nat.lt_or_ge r2.pos r1.pos with
                | Or.inl h => h
                | Or.inr hge21 => False.elim (hp_ne (Nat.le_antisymm hge21 hge12))
              have hell2 : r2.len ≤ mPred + 1 := Nat.le_trans (hWidths r2 hr2) hMod
              have hdis : r2.pos + r2.len ≤ r1.pos :=
                modPos_disjoint_intervals mPred r2.pos r1.pos r2.len hmod_eq hlt21 hell2
              Nat.not_le_of_gt hj_lt2 (Nat.le_trans hdis hp1_le)
      Or.inr (fun j =>
        match h2 : inWin r2.pos r2.len j.val with
        | true =>
            have hn1 : ¬(inWin r1.pos r1.len j.val = true) := hDisj j h2
            have hw1_j : applyWin r1 w0 j = w0 j := if_neg hn1
            have hw2_j : applyWin r2 (applyWin r1 w0) j = w0 j :=
              Eq.trans (if_pos h2) (hrhs2 j h2)
            Eq.trans hw1_j (Eq.symm hw2_j)
        | false =>
            have hn2 : ¬(inWin r2.pos r2.len j.val = true) :=
              fun ht => Bool.noConfusion (Eq.trans (Eq.symm h2) ht)
            Eq.symm (if_neg hn2))

/-- Core Lemma A2 (PR #348 Mask Guard Safety): Unconditionally (for ANY modulus mPred + 1),
    any concrete 2-step sequence whose second step satisfies `pr348MaskSuppressBool = true`
    returns `w0`! -/
theorem pr348_suppressed_pass2_returns_root {N : Nat}
    (R : List (WindowRule N)) (w0 : WordF N) (mPred : Nat)
    {r1 r2 : WindowRule N}
    (hr1 : r1 ∈ R) (hm1 : MatchesWin r1 w0)
    (hSup : pr348MaskSuppressBool R w0 mPred (modPos mPred r1.pos) r2 = true) :
    WordEq w0 (applyWin r2 (applyWin r1 w0)) :=
  let ⟨_, hb_rhs, hall_j⟩ := and3_eq_true hSup
  have hrhs2 : ∀ j : Fin N, inWin r2.pos r2.len j.val = true → r2.rhs j = w0 j := rhsAgrees_of_bool hb_rhs
  fun j =>
    match h2 : inWin r2.pos r2.len j.val with
    | true => Eq.symm (Eq.trans (if_pos h2) (hrhs2 j h2))
    | false =>
        have hn2 : ¬(inWin r2.pos r2.len j.val = true) :=
          fun ht => Bool.noConfusion (Eq.trans (Eq.symm h2) ht)
        have hw1_eq_w0 : applyWin r1 w0 j = w0 j :=
          match decEq (applyWin r1 w0 j) (w0 j) with
          | isTrue heq => heq
          | isFalse hne =>
              have hj_chk := finAll_true_forall hall_j j
              have hm1_b : matchesWinBool r1 w0 = true := matchesWinBool_of_matchesWin hm1
              have hmod_b : decide (modPos mPred r1.pos = modPos mPred r1.pos) = true := decide_eq_true rfl
              have hne_b : decide (applyWin r1 w0 j ≠ w0 j) = true := decide_eq_true hne
              have hr1_hit :
                  (matchesWinBool r1 w0 && decide (modPos mPred r1.pos = modPos mPred r1.pos) &&
                    decide (applyWin r1 w0 j ≠ w0 j)) = true :=
                Eq.trans
                  (congrArg (fun b => b && decide (applyWin r1 w0 j ≠ w0 j))
                    (Eq.trans (congrArg (fun x => x && _) hm1_b) hmod_b))
                  hne_b
              have hany : R.any (fun rx =>
                  matchesWinBool rx w0 && decide (modPos mPred rx.pos = modPos mPred r1.pos) &&
                    decide (applyWin rx w0 j ≠ w0 j)) = true :=
                listAny_of_mem_true hr1 hr1_hit
              have hin : inWin r2.pos r2.len j.val = true :=
                Eq.trans (Eq.symm (if_pos hany)) hj_chk
              False.elim (hn2 hin)
        Eq.symm (Eq.trans (if_neg hn2) hw1_eq_w0)

-- Congruence-Fibered Abstract Domain D^{(k)} = (base, fib_0 .. fib_{mPred})
structure ProvAdj2State (L : Nat) where
  base : Abs2 L
  fib  : Nat → Abs2 L

def gammaProv {L : Nat} (mPred : Nat) (D : ProvAdj2State L) (w : WordF (L + 1)) : Prop :=
  gamma2 D.base w ∨ ∃ f, f ≤ mPred ∧ gamma2 (D.fib f) w

def Pass1FiberWords {N : Nat} (R : List (WindowRule N)) (w0 : WordF N)
    (mPred f : Nat) (w1 : WordF N) : Prop :=
  ∃ r1, r1 ∈ R ∧ MatchesWin r1 w0 ∧ modPos mPred r1.pos = f ∧ WordEq (applyWin r1 w0) w1

def runGuardedProvAdj2 {L : Nat}
    (R : List (WindowRule (L + 1))) (w0 : WordF (L + 1)) (mPred : Nat)
    (pass2Suppress : Nat → WindowRule (L + 1) → Bool) : Nat → ProvAdj2State L
  | 0 =>
      { base := alpha2 (WordEq w0)
        fib  := fun _ _ _ _ => False }
  | 1 =>
      { base := alpha2 (WordEq w0)
        fib  := fun f => trellisReduce2 (alpha2 (Pass1FiberWords R w0 mPred f)) }
  | 2 =>
      let prev := runGuardedProvAdj2 R w0 mPred pass2Suppress 1
      { base := prev.base
        fib  := fun f => LocalAdj2GuardedStep R (pass2Suppress f) (prev.fib f) }
  | k + 3 =>
      let prev := runGuardedProvAdj2 R w0 mPred pass2Suppress (k + 2)
      { base := prev.base
        fib  := fun f => LocalAdj2GuardedStep R (fun _ => false) (prev.fib f) }

theorem runGuardedProvAdj2_base_eq {L : Nat}
    (R : List (WindowRule (L + 1))) (w0 : WordF (L + 1)) (mPred : Nat)
    (pass2Suppress : Nat → WindowRule (L + 1) → Bool) :
    ∀ k : Nat, (runGuardedProvAdj2 R w0 mPred pass2Suppress k).base = alpha2 (WordEq w0)
  | 0 => rfl
  | 1 => rfl
  | 2 => rfl
  | k + 3 => runGuardedProvAdj2_base_eq R w0 mPred pass2Suppress (k + 2)

theorem runGuardedProvAdj2_fib_mono_from_1 {L : Nat}
    (R : List (WindowRule (L + 1))) (w0 : WordF (L + 1)) (mPred : Nat)
    (pass2Suppress : Nat → WindowRule (L + 1) → Bool) (f : Nat) :
    ∀ (m : Nat) {w : WordF (L + 1)},
      gamma2 ((runGuardedProvAdj2 R w0 mPred pass2Suppress 1).fib f) w →
      gamma2 ((runGuardedProvAdj2 R w0 mPred pass2Suppress (m + 1)).fib f) w
  | 0, _, hw => hw
  | 1, _, hw => localAdj2GuardedStep_refl R (pass2Suppress f) hw
  | m + 2, _, hw =>
      localAdj2GuardedStep_refl R (fun _ => false)
        (runGuardedProvAdj2_fib_mono_from_1 R w0 mPred pass2Suppress f (m + 1) hw)

theorem pass1_concrete_decomp {L : Nat}
    (R : List (WindowRule (L + 1))) (w0 : WordF (L + 1)) (mPred : Nat)
    {w1 : WordF (L + 1)} (hw1 : IterConcreteWin R 1 (WordEq w0) w1) :
    WordEq w0 w1 ∨ ∃ f, f ≤ mPred ∧ Pass1FiberWords R w0 mPred f w1 :=
  match hw1 with
  | Or.inl h0 => Or.inl h0
  | Or.inr ⟨_, hu0, r1, hr1, hm1, heq1⟩ =>
      have hm1_w0 : MatchesWin r1 w0 := matchesWin_wordEq (wordEq_symm hu0) hm1
      have heq_w0 : WordEq (applyWin r1 w0) w1 :=
        wordEq_trans (applyWin_wordEq r1 hu0) heq1
      Or.inr ⟨modPos mPred r1.pos, modPos_le mPred r1.pos, r1, hr1, hm1_w0, rfl, heq_w0⟩

theorem pass1_word_in_fib {L : Nat}
    (R : List (WindowRule (L + 1))) (w0 : WordF (L + 1)) (mPred : Nat)
    (pass2Suppress : Nat → WindowRule (L + 1) → Bool)
    {f : Nat} {w1 : WordF (L + 1)} (hw1 : Pass1FiberWords R w0 mPred f w1) :
    gamma2 ((runGuardedProvAdj2 R w0 mPred pass2Suppress 1).fib f) w1 :=
  gamma2_trellisReduce2 (fun _ => ⟨w1, hw1, rfl, rfl⟩)

/-- Master Guarded Soundness Schema for `runGuardedProvAdj2`: any executable Boolean Pass-2
    guard satisfying the root-or-stay property makes the multi-pass congruence-fibered `Adj2`
    abstract interpreter sound for all `k : Nat` (zero axioms)! -/
theorem guarded_prov_adj2_soundness_of_pass2_safe {L : Nat} (hL : 0 < L)
    (R : List (WindowRule (L + 1))) (w0 : WordF (L + 1)) (mPred : Nat)
    (pass2Suppress : Nat → WindowRule (L + 1) → Bool)
    (hSafe : ∀ {r1 r2 : WindowRule (L + 1)},
      r1 ∈ R → MatchesWin r1 w0 →
      r2 ∈ R → MatchesWin r2 (applyWin r1 w0) →
      pass2Suppress (modPos mPred r1.pos) r2 = true →
      WordEq w0 (applyWin r2 (applyWin r1 w0)) ∨
      WordEq (applyWin r1 w0) (applyWin r2 (applyWin r1 w0))) :
    ∀ (k : Nat) (w : WordF (L + 1)),
      IterConcreteWin R k (WordEq w0) w →
      gammaProv mPred (runGuardedProvAdj2 R w0 mPred pass2Suppress k) w
  | 0, w, hw0 =>
      Or.inl (fun _ => ⟨w, hw0, rfl, rfl⟩)
  | 1, w, hw1 =>
      match pass1_concrete_decomp R w0 mPred hw1 with
      | Or.inl h0 => Or.inl (fun _ => ⟨w, h0, rfl, rfl⟩)
      | Or.inr ⟨f, hf, hpf⟩ => Or.inr ⟨f, hf, pass1_word_in_fib R w0 mPred pass2Suppress hpf⟩
  | 2, w2, hw2 =>
      match hw2 with
      | Or.inl hw1 =>
          match pass1_concrete_decomp R w0 mPred hw1 with
          | Or.inl h0 => Or.inl (fun _ => ⟨w2, h0, rfl, rfl⟩)
          | Or.inr ⟨f, hf, hpf⟩ =>
              Or.inr ⟨f, hf,
                runGuardedProvAdj2_fib_mono_from_1 R w0 mPred pass2Suppress f 1
                  (pass1_word_in_fib R w0 mPred pass2Suppress hpf)⟩
      | Or.inr ⟨w1, hw1, r2, hr2, hm2, heq2⟩ =>
          match pass1_concrete_decomp R w0 mPred hw1 with
          | Or.inl hw0_w1 =>
              have hm2_w0 : MatchesWin r2 w0 := matchesWin_wordEq (wordEq_symm hw0_w1) hm2
              have heq_w0 : WordEq (applyWin r2 w0) w2 :=
                wordEq_trans (applyWin_wordEq r2 hw0_w1) heq2
              let f2 := modPos mPred r2.pos
              have hpf2 : Pass1FiberWords R w0 mPred f2 w2 := ⟨r2, hr2, hm2_w0, rfl, heq_w0⟩
              Or.inr ⟨f2, modPos_le mPred r2.pos,
                runGuardedProvAdj2_fib_mono_from_1 R w0 mPred pass2Suppress f2 1
                  (pass1_word_in_fib R w0 mPred pass2Suppress hpf2)⟩
          | Or.inr ⟨f, hf, ⟨r1, hr1, hm1, hf_eq, heq1⟩⟩ =>
              let u1 := applyWin r1 w0
              have hm2_u1 : MatchesWin r2 u1 := matchesWin_wordEq (wordEq_symm heq1) hm2
              have heq2_u1 : WordEq (applyWin r2 u1) w2 :=
                wordEq_trans (applyWin_wordEq r2 heq1) heq2
              have hw1_in_D1 : gamma2 ((runGuardedProvAdj2 R w0 mPred pass2Suppress 1).fib f) w1 :=
                pass1_word_in_fib R w0 mPred pass2Suppress ⟨r1, hr1, hm1, hf_eq, heq1⟩
              match hsup : pass2Suppress f r2 with
              | false =>
                  Or.inr ⟨f, hf,
                    localAdj2_unsuppressed_firing_sound R (pass2Suppress f) hw1_in_D1 hr2 hsup hm2 heq2⟩
              | true =>
                  have hsup_r1 : pass2Suppress (modPos mPred r1.pos) r2 = true :=
                    Eq.subst (motive := fun x => pass2Suppress x r2 = true) (Eq.symm hf_eq) hsup
                  match hSafe hr1 hm1 hr2 hm2_u1 hsup_r1 with
                  | Or.inl hw0_u2 =>
                      have hw0_w2 : WordEq w0 w2 := wordEq_trans hw0_u2 heq2_u1
                      Or.inl (fun _ => ⟨w2, hw0_w2, rfl, rfl⟩)
                  | Or.inr hu1_u2 =>
                      have hw1_w2 : WordEq w1 w2 :=
                        wordEq_trans (wordEq_symm heq1) (wordEq_trans hu1_u2 heq2_u1)
                      have hw2_in_D1 : gamma2 ((runGuardedProvAdj2 R w0 mPred pass2Suppress 1).fib f) w2 :=
                        gamma2_wordEq hw1_w2 hw1_in_D1
                      Or.inr ⟨f, hf,
                        runGuardedProvAdj2_fib_mono_from_1 R w0 mPred pass2Suppress f 1 hw2_in_D1⟩
  | k + 3, w', hw' =>
      have ih := guarded_prov_adj2_soundness_of_pass2_safe hL R w0 mPred pass2Suppress hSafe (k + 2)
      match hw' with
      | Or.inl hw =>
          match ih w' hw with
          | Or.inl hbase =>
              have hbe := runGuardedProvAdj2_base_eq R w0 mPred pass2Suppress (k + 2)
              have hbe3 := runGuardedProvAdj2_base_eq R w0 mPred pass2Suppress (k + 3)
              have h0 : gamma2 (alpha2 (WordEq w0)) w' :=
                Eq.subst (motive := fun b => gamma2 b w') hbe hbase
              Or.inl (Eq.subst (motive := fun b => gamma2 b w') (Eq.symm hbe3) h0)
          | Or.inr ⟨f, hf, hfib⟩ =>
              Or.inr ⟨f, hf, localAdj2GuardedStep_refl R (fun _ => false) hfib⟩
      | Or.inr ⟨w, hw, r, hr, hm, heq⟩ =>
          match ih w hw with
          | Or.inl hbase =>
              have hbe := runGuardedProvAdj2_base_eq R w0 mPred pass2Suppress (k + 2)
              have h0 : gamma2 (alpha2 (WordEq w0)) w :=
                Eq.subst (motive := fun b => gamma2 b w) hbe hbase
              have hw0_w : WordEq w0 w := gamma2_singleton_exact hL w0 w h0
              have hm_w0 : MatchesWin r w0 := matchesWin_wordEq (wordEq_symm hw0_w) hm
              have heq_w0 : WordEq (applyWin r w0) w' :=
                wordEq_trans (applyWin_wordEq r hw0_w) heq
              let fr := modPos mPred r.pos
              have hpf : Pass1FiberWords R w0 mPred fr w' := ⟨r, hr, hm_w0, rfl, heq_w0⟩
              Or.inr ⟨fr, modPos_le mPred r.pos,
                runGuardedProvAdj2_fib_mono_from_1 R w0 mPred pass2Suppress fr (k + 2)
                  (pass1_word_in_fib R w0 mPred pass2Suppress hpf)⟩
          | Or.inr ⟨f, hf, hfib⟩ =>
              Or.inr ⟨f, hf,
                localAdj2_unsuppressed_firing_sound R (fun _ => false) hfib hr rfl hm heq⟩

/-- THEOREM A (Fable's Guarded Congruence-Fibered Adj2 Soundness Theorem, Zero Axioms):
    For every word length `L + 1 >= 2`, every window rewrite system `R` with rule lengths
    in `[2, ellMax]`, every modulus `m = mPred + 1 >= ellMax`, every root word `w0`,
    and every pass horizon `k >= 0`, concrete `<= k`-step reachability is contained in
    the concretization of `runGuardedProvAdj2` under `fableLenSuppressBool`! -/
theorem theorem_A_fable_guarded_prov_adj2_soundness {L : Nat} (hL : 0 < L)
    (R : List (WindowRule (L + 1))) (w0 : WordF (L + 1)) (mPred ellMax : Nat)
    (hMod : ellMax ≤ mPred + 1)
    (hWidths : ∀ r, r ∈ R → r.len ≤ ellMax)
    (k : Nat) (w : WordF (L + 1))
    (hReach : IterConcreteWin R k (WordEq w0) w) :
    gammaProv mPred (runGuardedProvAdj2 R w0 mPred (fableLenSuppressBool R w0 mPred) k) w :=
  guarded_prov_adj2_soundness_of_pass2_safe hL R w0 mPred (fableLenSuppressBool R w0 mPred)
    (fun hr1 hm1 hr2 hm2 hsup =>
      fable_suppressed_pass2_returns_root_or_stays R w0 mPred ellMax hMod hWidths hr1 hm1 hr2 hm2 hsup)
    k w hReach

/-- THEOREM A' (PR #348 Conservative Changed-Coordinate Mask Guard Soundness, Zero Axioms):
    Unconditionally (for ANY modulus `m = mPred + 1 >= 1` and ANY maximum rule length),
    concrete `<= k`-step reachability is contained in the concretization of
    `runGuardedProvAdj2` under `pr348MaskSuppressBool`! -/
theorem theorem_A_pr348_mask_guarded_prov_adj2_soundness {L : Nat} (hL : 0 < L)
    (R : List (WindowRule (L + 1))) (w0 : WordF (L + 1)) (mPred : Nat)
    (k : Nat) (w : WordF (L + 1))
    (hReach : IterConcreteWin R k (WordEq w0) w) :
    gammaProv mPred (runGuardedProvAdj2 R w0 mPred (pr348MaskSuppressBool R w0 mPred) k) w :=
  guarded_prov_adj2_soundness_of_pass2_safe hL R w0 mPred (pr348MaskSuppressBool R w0 mPred)
    (fun hr1 hm1 _ _ hsup =>
      Or.inl (pr348_suppressed_pass2_returns_root R w0 mPred hr1 hm1 hsup))
    k w hReach

-- ============================================================================
-- Section 5. Kernel-Verified Referee Counterexamples & Guard Comparisons on Adj2
-- ============================================================================

-- Executable Guarded `prov_{Fp} x adj2` Interpreters (`fable_len_guard` vs `pr348_mask_guard`)
def hasLongerPass1At (ors : List SubRule) (w0 : WordL) (p ell : Nat) : Bool :=
  (neighborsWord ors w0).any (fun (p1, r1, _) => (p1 == p) && decide (ell < r1.lhs.length))

def fiberChangedCoords (ors : List SubRule) (w0 : WordL) (fp fIdx : Nat) : List Nat :=
  let step1 := (neighborsWord ors w0).filter (fun (p1, _, _) => p1 % fp == fIdx)
  (List.range w0.length).filter (fun i =>
    step1.any (fun (_, _, w1) => (w1.drop i).head? != (w0.drop i).head?))

def adj2TransferStepFableGuard (lm1 : Nat) (ors : List SubRule) (w0 : WordL)
    (fp fIdx : Nat) (isPass2 : Bool) (arcs : List BigramArc) : List BigramArc :=
  let ta := trellisFilterAdj2 lm1 arcs
  let positions := List.range lm1
  let added := ors.foldr (fun r acc =>
    (positions.foldr (fun p acc2 =>
      let skip := isPass2 && (p % fp == fIdx) &&
        (p + r.rhs.length ≤ w0.length && listSliceEq (w0.drop p) r.rhs) &&
        !hasLongerPass1At ors w0 p r.rhs.length
      if skip then acc2 else adj2StepRuleAt lm1 ta r p ++ acc2) []) ++ acc) []
  trellisFilterAdj2 lm1 (dedupArcs (ta ++ added))

def adj2TransferStepMaskGuard (lm1 : Nat) (ors : List SubRule) (w0 : WordL)
    (fp fIdx : Nat) (changed : List Nat) (isPass2 : Bool) (arcs : List BigramArc) : List BigramArc :=
  let ta := trellisFilterAdj2 lm1 arcs
  let positions := List.range lm1
  let added := ors.foldr (fun r acc =>
    (positions.foldr (fun p acc2 =>
      let subsetMask := changed.all (fun i => decide (p ≤ i) && decide (i < p + r.rhs.length))
      let skip := isPass2 && (p % fp == fIdx) &&
        (p + r.rhs.length ≤ w0.length && listSliceEq (w0.drop p) r.rhs) && subsetMask
      if skip then acc2 else adj2StepRuleAt lm1 ta r p ++ acc2) []) ++ acc) []
  trellisFilterAdj2 lm1 (dedupArcs (ta ++ added))

def runFableGuardedProvFpAdj2 (ors : List SubRule) (w0 : WordL) (fp kMax : Nat) : List (List BigramArc) :=
  let lm1 := w0.length - 1
  let base := wordToAdj2 w0
  match kMax with
  | 0 => [base]
  | k + 1 =>
      let step1 := neighborsWord ors w0
      let initFibs := (List.range fp).map (fun f =>
        let matching := step1.filter (fun (p, _, _) => p % fp == f)
        let raw := matching.foldr (fun (_, _, w1) acc => wordToAdj2 w1 ++ acc) []
        if matching.isEmpty then [] else trellisFilterAdj2 lm1 (dedupArcs raw))
      let rec loopSteps : Nat → Nat → List (List BigramArc) → List (List BigramArc)
        | 0, _, fibs => fibs
        | m + 1, stepIdx, fibs =>
            let nextFibs := (List.range fp).zip fibs |>.map (fun (f, fa) =>
              if fa.isEmpty then []
              else adj2TransferStepFableGuard lm1 ors w0 fp f (stepIdx == 1) fa)
            loopSteps m (stepIdx + 1) nextFibs
      base :: loopSteps k 1 initFibs

def runMaskGuardedProvFpAdj2 (ors : List SubRule) (w0 : WordL) (fp kMax : Nat) : List (List BigramArc) :=
  let lm1 := w0.length - 1
  let base := wordToAdj2 w0
  match kMax with
  | 0 => [base]
  | k + 1 =>
      let step1 := neighborsWord ors w0
      let chgByFib := (List.range fp).map (fun f => fiberChangedCoords ors w0 fp f)
      let initFibs := (List.range fp).map (fun f =>
        let matching := step1.filter (fun (p, _, _) => p % fp == f)
        let raw := matching.foldr (fun (_, _, w1) acc => wordToAdj2 w1 ++ acc) []
        if matching.isEmpty then [] else trellisFilterAdj2 lm1 (dedupArcs raw))
      let rec loopSteps : Nat → Nat → List (List BigramArc) → List (List BigramArc)
        | 0, _, fibs => fibs
        | m + 1, stepIdx, fibs =>
            let nextFibs := ((List.range fp).zip chgByFib).zip fibs |>.map (fun ((f, chg), fa) =>
              if fa.isEmpty then []
              else adj2TransferStepMaskGuard lm1 ors w0 fp f chg (stepIdx == 1) fa)
            loopSteps m (stepIdx + 1) nextFibs
      base :: loopSteps k 1 initFibs

def wordAcceptedByFibs (lm1 : Nat) (fibs : List (List BigramArc)) (cand : WordL) : Bool :=
  fibs.any (fun fa => !fa.isEmpty && wordInGamma2 lm1 fa cand)

-- 5.1 Referee Counterexample R* = {abc <-> def, ab <-> de, bf <-> cc} on w* = "abc"
def wRStar : WordL := [.a, .b, .c]
def rulesRStar : List SubRule := orientRules [
  ⟨[.a, .b, .c], [.d, .e, .f]⟩,
  ⟨[.a, .b], [.d, .e]⟩,
  ⟨[.b, .f], [.c, .c]⟩
]

/-- Theorem 15 (Referee Counterexample R* on Executable Adj2 Fibers):
    - Concrete reachability reaches `abf` at k = 2 and `acc` (symbol `c` at pos 1) at k = 3.
    - Legacy unguarded suppression loses `abf` at k = 2 and loses `acc` / `(1, c)` at k = 3.
    - Both Fable's length guard and PR #348's mask guard retain `abf` at k = 2 and `acc` at k = 3! -/
theorem referee_rstar_legacy_unsound_and_both_guards_sound :
    (concreteIterWords rulesRStar 2 [wRStar]).elem [.a, .b, .f] = true ∧
    (concreteIterWords rulesRStar 3 [wRStar]).elem [.a, .c, .c] = true ∧
    wordAcceptedByProvFpAdj2 rulesRStar wRStar 4 2 [.a, .b, .f] = false ∧
    wordAcceptedByProvFpAdj2 rulesRStar wRStar 4 3 [.a, .c, .c] = false ∧
    provFpHasSymbolAtPos1 rulesRStar wRStar 4 3 .c = false ∧
    wordAcceptedByFibs 2 (runFableGuardedProvFpAdj2 rulesRStar wRStar 4 2) [.a, .b, .f] = true ∧
    wordAcceptedByFibs 2 (runFableGuardedProvFpAdj2 rulesRStar wRStar 4 3) [.a, .c, .c] = true ∧
    wordAcceptedByFibs 2 (runMaskGuardedProvFpAdj2 rulesRStar wRStar 4 2) [.a, .b, .f] = true ∧
    wordAcceptedByFibs 2 (runMaskGuardedProvFpAdj2 rulesRStar wRStar 4 3) [.a, .c, .c] = true :=
  ⟨rfl, ⟨rfl, ⟨rfl, ⟨rfl, ⟨rfl, ⟨rfl, ⟨rfl, ⟨rfl, rfl⟩⟩⟩⟩⟩⟩⟩⟩

-- 5.2 LBF-17 Counterexample 1 Precision Preserved Under Both Repaired Guards
def rulesCex1 : List SubRule := orientRules [
  ⟨[.a, .b], [.d, .e]⟩,
  ⟨[.b, .c], [.f, .a]⟩,
  ⟨[.d, .f], [.c, .c]⟩
]

theorem lbf17_cex1_chimera_excluded_under_both_guards :
    (concreteIterWords rulesCex1 4 [wRStar]).elem [.c, .c, .a] = false ∧
    (runFableGuardedProvFpAdj2 rulesCex1 wRStar 4 4).any
      (fun fa => fa.any (fun (p, x, _) => p == 0 && x == .c)) = false ∧
    (runMaskGuardedProvFpAdj2 rulesCex1 wRStar 4 4).any
      (fun fa => fa.any (fun (p, x, _) => p == 0 && x == .c)) = false :=
  ⟨rfl, ⟨rfl, rfl⟩⟩

-- 5.3 Necessity of `Fp >= ell_max` for `fable_len_guard` vs Unconditional `pr348_mask_guard`:
--     w0 = "abcd", rules {abc <-> efd, dd <-> cd}, ell_max = 3.
--     When Fp = 2 < 3, `abc -> efd` at p=0 (0%2=0) yields "efdd"; at Pass 2, `dd -> cd` at p=2 (2%2=0)
--     has NO longer Pass-1 redex at p=2, so `fable_len_guard` at Fp=2 wrongly suppresses it and loses "efcd",
--     whereas `fable_len_guard` at Fp=3 >= ell_max AND `pr348_mask_guard` at Fp=2 both retain "efcd"!
def wModNeed : WordL := [.a, .b, .c, .d]
def rulesModNeed : List SubRule := orientRules [
  ⟨[.a, .b, .c], [.e, .f, .d]⟩,
  ⟨[.d, .d], [.c, .d]⟩
]

theorem referee_modulus_assumption_necessary_for_len_guard :
    (concreteIterWords rulesModNeed 2 [wModNeed]).elem [.e, .f, .c, .d] = true ∧
    wordAcceptedByFibs 3 (runFableGuardedProvFpAdj2 rulesModNeed wModNeed 2 2) [.e, .f, .c, .d] = false ∧
    wordAcceptedByFibs 3 (runFableGuardedProvFpAdj2 rulesModNeed wModNeed 3 2) [.e, .f, .c, .d] = true ∧
    wordAcceptedByFibs 3 (runMaskGuardedProvFpAdj2 rulesModNeed wModNeed 2 2) [.e, .f, .c, .d] = true :=
  ⟨rfl, ⟨rfl, ⟨rfl, rfl⟩⟩⟩

-- 5.4 Modulo-Collision Guard Comparison (`w0 = "abccab"`, `ab <-> de`, `Fp = 4 >= ell_max = 2`):
--     (1) `fable_len_guard` suppresses `de -> ab` at p=0 and p=4 (`!hasLongerPass1At = true`),
--         whereas `pr348_mask_guard` declines to suppress (`D_0 = [0,1,4,5]` not subset of `[0,2)`).
--     (2) Yet on `Adj2` fibers when `Fp >= ell_max`, because `"abccde"` and `"deccab"` already
--         supply `w0`'s disjoint slices in Fiber 0 at Pass 1, BOTH guards induce identical `Adj2` states!
def wModColl : WordL := [.a, .b, .c, .c, .a, .b]
def rulesModColl : List SubRule := orientRules [
  ⟨[.a, .b], [.d, .e]⟩
]

theorem referee_mod_collision_len_guard_sharper_than_mask_guard :
    (!hasLongerPass1At rulesModColl wModColl 0 2) = true ∧
    ((fiberChangedCoords rulesModColl wModColl 4 0).all (fun i => decide (0 ≤ i) && decide (i < 2))) = false ∧
    (runFableGuardedProvFpAdj2 rulesModColl wModColl 4 3 ==
     runMaskGuardedProvFpAdj2 rulesModColl wModColl 4 3) = true :=
  ⟨rfl, ⟨rfl, rfl⟩⟩

-- 5.5 Referee CE-5: Single-Rule Horizon Overshoot (`w0 = "abeeab"`, `ab <-> cd`, `Fp = 4`):
--     `"cdeecd"` is at concrete distance 2, yet appears in Fiber 0 at Pass 1.
def wCe5 : WordL := [.a, .b, .e, .e, .a, .b]
def rulesCe5 : List SubRule := orientRules [⟨[.a, .b], [.c, .d]⟩]

theorem referee_ce5_single_rule_horizon_overshoot :
    (concreteIterWords rulesCe5 1 [wCe5]).elem [.c, .d, .e, .e, .c, .d] = false ∧
    (concreteIterWords rulesCe5 2 [wCe5]).elem [.c, .d, .e, .e, .c, .d] = true ∧
    wordAcceptedByFibs 5 (runFableGuardedProvFpAdj2 rulesCe5 wCe5 4 1) [.c, .d, .e, .e, .c, .d] = true :=
  ⟨rfl, ⟨rfl, rfl⟩⟩

end LBF18IncrementalAndCollisions

#print axioms LBF18IncrementalAndCollisions.keyed_cache_lookup_sound
#print axioms LBF18IncrementalAndCollisions.keyed_cache_rejects_mismatch
#print axioms LBF18IncrementalAndCollisions.incremental_word_map_equiv
#print axioms LBF18IncrementalAndCollisions.incremental_pair_map_equiv
#print axioms LBF18IncrementalAndCollisions.zero_preserving_gate_support
#print axioms LBF18IncrementalAndCollisions.shapley_sum_eq_twice_joint
#print axioms LBF18IncrementalAndCollisions.marginal_sum_minus_joint_eq_interaction
#print axioms LBF18IncrementalAndCollisions.mod_collision_disjoint_intervals
#print axioms LBF18IncrementalAndCollisions.disjoint_mod4_rewrites_commute_in_2_steps
#print axioms LBF18IncrementalAndCollisions.cex4a_fp4_spurious_fp5_exact
#print axioms LBF18IncrementalAndCollisions.cex4b_intrafiber_step2_fork
#print axioms LBF18IncrementalAndCollisions.cex4c_horizon_compression_long_word
#print axioms LBF18IncrementalAndCollisions.localAdj2_unsuppressed_firing_sound
#print axioms LBF18IncrementalAndCollisions.fable_suppressed_pass2_returns_root_or_stays
#print axioms LBF18IncrementalAndCollisions.pr348_suppressed_pass2_returns_root
#print axioms LBF18IncrementalAndCollisions.theorem_A_fable_guarded_prov_adj2_soundness
#print axioms LBF18IncrementalAndCollisions.theorem_A_pr348_mask_guarded_prov_adj2_soundness
#print axioms LBF18IncrementalAndCollisions.referee_rstar_legacy_unsound_and_both_guards_sound
#print axioms LBF18IncrementalAndCollisions.lbf17_cex1_chimera_excluded_under_both_guards
#print axioms LBF18IncrementalAndCollisions.referee_modulus_assumption_necessary_for_len_guard
#print axioms LBF18IncrementalAndCollisions.referee_mod_collision_len_guard_sharper_than_mask_guard
#print axioms LBF18IncrementalAndCollisions.referee_ce5_single_rule_horizon_overshoot
