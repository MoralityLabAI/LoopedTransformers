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
