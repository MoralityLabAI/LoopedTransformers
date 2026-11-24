/-
  LBF-17: Relational Abstract Interpretation and Provenance-Preserving Frescoes.
  Standalone Lean 4.19.0 formalization (zero external dependencies, zero axioms).

  Contents:
  1. Constructive Galois connections (α, γ), monotonicity, and k-step abstract
     transfer soundness (repaired LBF-16 horizon-support overapproximation theorem).
  2. Relational fresco hierarchy on fixed-length words (Fin L → Sym):
     - Positionwise 1-gram support (α₁, γ₁) and multiplicative window transfer soundness
     - Adjacent 2-gram support (α₂, γ₂)
     - Length-3 subterm support (α₃, γ₃)
     - Branch-provenance fiber separation and non-backtracking root decomposition
     - Strict precision ordering: γ₃ ∘ α₃ ⊆ γ₂ ∘ α₂ ⊆ γ₁ ∘ α₁
  3. Kernel-verified counterexamples (proved by definitional reduction Eq.refl):
     - Counterexample 1 (w₀ = "abc", {ab↔de, bc↔fa, df↔cc}):
       Refutes LBF-16 pass-2 exactness; proves 1-step backtracking provenance delays the
       chimera to pass 4 (via w₁→w₀→w₂→chimera), while 2-gram, 3-gram, and non-backtracking
       provenance remain exact across all passes.
     - Counterexample 2 (w₀ = "abbc", {ab↔de, bbc↔efa, def↔ccc}):
       Refutes 2-gram pass-2 exactness via shared-middle overlap; 3-gram and non-backtracking
       provenance remain exact.
     - Counterexample 3 (w₀ = "aebcd", {aeb↔def, bcd↔ffa, ff↔cc, dec↔bbb}):
       Refutes 3-gram global consistency (spurious chimera "deffa" → "decca" → "bbb");
       non-backtracking branch-provenance fibers exclude all cross-branch chimeras.
-/

namespace LBF17AbstractInterp

-- ============================================================================
-- Section 1. Constructive Galois Connections & k-Step Soundness Theorem
-- ============================================================================

def SubsetPred {W : Type} (P Q : W → Prop) : Prop :=
  ∀ w, P w → Q w

theorem subset_refl {W : Type} (P : W → Prop) : SubsetPred P P :=
  fun _ h => h

theorem subset_trans {W : Type} {P Q R : W → Prop}
    (hPQ : SubsetPred P Q) (hQR : SubsetPred Q R) : SubsetPred P R :=
  fun w hw => hQR w (hPQ w hw)

structure GaloisConn {W A : Type}
    (α : (W → Prop) → A) (γ : A → (W → Prop)) (leA : A → A → Prop) : Prop where
  le_refl : ∀ a, leA a a
  le_trans : ∀ {a b c}, leA a b → leA b c → leA a c
  adj : ∀ S a, leA (α S) a ↔ SubsetPred S (γ a)

theorem gc_extensive {W A : Type} {α : (W → Prop) → A} {γ : A → (W → Prop)} {leA : A → A → Prop}
    (G : GaloisConn α γ leA) (S : W → Prop) : SubsetPred S (γ (α S)) :=
  (G.adj S (α S)).mp (G.le_refl (α S))

theorem gc_reductive {W A : Type} {α : (W → Prop) → A} {γ : A → (W → Prop)} {leA : A → A → Prop}
    (G : GaloisConn α γ leA) (a : A) : leA (α (γ a)) a :=
  (G.adj (γ a) a).mpr (subset_refl (γ a))

theorem gc_mono_alpha {W A : Type} {α : (W → Prop) → A} {γ : A → (W → Prop)} {leA : A → A → Prop}
    (G : GaloisConn α γ leA) {S T : W → Prop} (hST : SubsetPred S T) : leA (α S) (α T) :=
  (G.adj S (α T)).mpr (subset_trans hST (gc_extensive G T))

theorem gc_mono_gamma {W A : Type} {α : (W → Prop) → A} {γ : A → (W → Prop)} {leA : A → A → Prop}
    (G : GaloisConn α γ leA) {a b : A} (hab : leA a b) : SubsetPred (γ a) (γ b) :=
  (G.adj (γ a) b).mp (G.le_trans (gc_reductive G a) hab)

def IterConcrete {W : Type} (F : (W → Prop) → (W → Prop)) : Nat → (W → Prop) → (W → Prop)
  | 0, S => S
  | k + 1, S => F (IterConcrete F k S)

def IterAbstract {A : Type} (Fs : A → A) : Nat → A → A
  | 0, a => a
  | k + 1, a => Fs (IterAbstract Fs k a)

/-- General k-step soundness theorem: any sound abstract transfer operator overapproximates
    concrete k-step reachability for all horizons k. -/
theorem abstract_iter_soundness {W A : Type}
    (F : (W → Prop) → (W → Prop))
    (hF_mono : ∀ {P Q : W → Prop}, SubsetPred P Q → SubsetPred (F P) (F Q))
    (γ : A → (W → Prop))
    (Fs : A → A)
    (h_step_sound : ∀ a : A, SubsetPred (F (γ a)) (γ (Fs a))) :
    ∀ (k : Nat) (S₀ : W → Prop) (a₀ : A),
      SubsetPred S₀ (γ a₀) →
      SubsetPred (IterConcrete F k S₀) (γ (IterAbstract Fs k a₀))
  | 0, _, _, h0 => h0
  | k + 1, S₀, a₀, h0 =>
      let ih := abstract_iter_soundness F hF_mono γ Fs h_step_sound k S₀ a₀ h0
      subset_trans (hF_mono ih) (h_step_sound (IterAbstract Fs k a₀))

/-- Best abstract transformer Fs := α ∘ F ∘ γ is automatically k-step sound. -/
theorem best_transformer_soundness {W A : Type}
    {α : (W → Prop) → A} {γ : A → (W → Prop)} {leA : A → A → Prop}
    (G : GaloisConn α γ leA)
    (F : (W → Prop) → (W → Prop))
    (hF_mono : ∀ {P Q : W → Prop}, SubsetPred P Q → SubsetPred (F P) (F Q))
    (k : Nat) (S₀ : W → Prop) :
    SubsetPred (IterConcrete F k S₀) (γ (IterAbstract (fun a => α (F (γ a))) k (α S₀))) :=
  abstract_iter_soundness F hF_mono γ (fun a => α (F (γ a)))
    (fun a => gc_extensive G (F (γ a)))
    k S₀ (α S₀) (gc_extensive G S₀)

-- ============================================================================
-- Section 2. Relational Hierarchy & Repaired Support Soundness on Fixed Words
-- ============================================================================

inductive Sym : Type where
  | a | b | c | d | e | f
  deriving DecidableEq, Repr

def WordF (L : Nat) : Type := Fin L → Sym

-- 2.1 Positionwise (1-gram) abstraction (LBF-16 envelope support domain)
def Abs1 (L : Nat) : Type := Fin L → Sym → Prop
def leAbs1 {L : Nat} (A B : Abs1 L) : Prop := ∀ p s, A p s → B p s
def alpha1 {L : Nat} (S : WordF L → Prop) : Abs1 L := fun p s => ∃ w, S w ∧ w p = s
def gamma1 {L : Nat} (A : Abs1 L) : WordF L → Prop := fun w => ∀ p, A p (w p)

theorem galois_pos1 (L : Nat) : GaloisConn (@alpha1 L) (@gamma1 L) (@leAbs1 L) where
  le_refl := fun _ _ _ h => h
  le_trans := fun h1 h2 p s h => h2 p s (h1 p s h)
  adj := fun _ _ =>
    Iff.intro
      (fun h w hw p => h p (w p) ⟨w, hw, rfl⟩)
      (fun h p _ ⟨w, hw, hwp⟩ => hwp ▸ h w hw p)

-- 2.2 Joint adjacent-symbol (2-gram) abstraction
def Abs2 (L : Nat) : Type := Fin L → Sym → Sym → Prop
def leAbs2 {L : Nat} (A B : Abs2 L) : Prop := ∀ p s0 s1, A p s0 s1 → B p s0 s1
def alpha2 {L : Nat} (S : WordF (L + 1) → Prop) : Abs2 L :=
  fun p s0 s1 => ∃ w, S w ∧ w p.castSucc = s0 ∧ w p.succ = s1
def gamma2 {L : Nat} (A : Abs2 L) : WordF (L + 1) → Prop :=
  fun w => ∀ p : Fin L, A p (w p.castSucc) (w p.succ)

theorem galois_adj2 (L : Nat) : GaloisConn (@alpha2 L) (@gamma2 L) (@leAbs2 L) where
  le_refl := fun _ _ _ _ h => h
  le_trans := fun h1 h2 p s0 s1 h => h2 p s0 s1 (h1 p s0 s1 h)
  adj := fun _ _ =>
    Iff.intro
      (fun h w hw p => h p (w p.castSucc) (w p.succ) ⟨w, hw, rfl, rfl⟩)
      (fun h p _ _ ⟨w, hw, h0, h1⟩ => h0 ▸ h1 ▸ h w hw p)

-- 2.3 Length-three subterm (3-gram) abstraction
def Abs3 (L : Nat) : Type := Fin L → Sym → Sym → Sym → Prop
def leAbs3 {L : Nat} (A B : Abs3 L) : Prop := ∀ p s0 s1 s2, A p s0 s1 s2 → B p s0 s1 s2
def alpha3 {L : Nat} (S : WordF (L + 2) → Prop) : Abs3 L :=
  fun p s0 s1 s2 =>
    ∃ w, S w ∧ w p.castSucc.castSucc = s0 ∧ w p.succ.castSucc = s1 ∧ w p.succ.succ = s2
def gamma3 {L : Nat} (A : Abs3 L) : WordF (L + 2) → Prop :=
  fun w => ∀ p : Fin L, A p (w p.castSucc.castSucc) (w p.succ.castSucc) (w p.succ.succ)

theorem galois_sub3 (L : Nat) : GaloisConn (@alpha3 L) (@gamma3 L) (@leAbs3 L) where
  le_refl := fun _ _ _ _ _ h => h
  le_trans := fun h1 h2 p s0 s1 s2 h => h2 p s0 s1 s2 (h1 p s0 s1 s2 h)
  adj := fun _ _ =>
    Iff.intro
      (fun h w hw p =>
        h p (w p.castSucc.castSucc) (w p.succ.castSucc) (w p.succ.succ) ⟨w, hw, rfl, rfl, rfl⟩)
      (fun h p _ _ _ ⟨w, hw, h0, h1, h2⟩ => h0 ▸ h1 ▸ h2 ▸ h w hw p)

-- 2.4 Constructive Precision Ordering Across the Relational Hierarchy (zero axioms)
private theorem fin_eq_of_val_eq {n : Nat} : ∀ {i j : Fin n}, i.val = j.val → i = j
  | ⟨_, _⟩, ⟨_, _⟩, rfl => rfl

theorem adj2_refines_pos1 {L : Nat} (_hL : 0 < L) (S : WordF (L + 1) → Prop) :
    SubsetPred (gamma2 (alpha2 S)) (gamma1 (alpha1 S)) :=
  fun w hw p =>
    match L, S, w, hw, p with
    | m + 1, _, w, hw, ⟨0, h0⟩ =>
        let q : Fin (m + 1) := ⟨0, Nat.zero_lt_succ m⟩
        have hqp : q.castSucc = ⟨0, h0⟩ := fin_eq_of_val_eq rfl
        match hw q with
        | ⟨w', hw', h1, _⟩ =>
            have heq : w' ⟨0, h0⟩ = w ⟨0, h0⟩ :=
              Eq.trans (congrArg w' (Eq.symm hqp)) (Eq.trans h1 (congrArg w hqp))
            ⟨w', hw', heq⟩
    | m + 1, _, w, hw, ⟨k + 1, hk⟩ =>
        let q : Fin (m + 1) := ⟨k, Nat.lt_of_succ_lt_succ hk⟩
        have hqp : q.succ = ⟨k + 1, hk⟩ := fin_eq_of_val_eq rfl
        match hw q with
        | ⟨w', hw', _, h2⟩ =>
            have heq : w' ⟨k + 1, hk⟩ = w ⟨k + 1, hk⟩ :=
              Eq.trans (congrArg w' (Eq.symm hqp)) (Eq.trans h2 (congrArg w hqp))
            ⟨w', hw', heq⟩

theorem sub3_refines_adj2 {L : Nat} (_hL : 0 < L) (S : WordF (L + 2) → Prop) :
    SubsetPred (gamma3 (alpha3 S)) (gamma2 (alpha2 S)) :=
  fun w hw p =>
    match L, S, w, hw, p with
    | m + 1, _, w, hw, ⟨0, h0⟩ =>
        let q : Fin (m + 1) := ⟨0, Nat.zero_lt_succ m⟩
        have hc0 : q.castSucc.castSucc = (⟨0, h0⟩ : Fin (m + 2)).castSucc := fin_eq_of_val_eq rfl
        have hc1 : q.succ.castSucc = (⟨0, h0⟩ : Fin (m + 2)).succ := fin_eq_of_val_eq rfl
        match hw q with
        | ⟨w', hw', e0, e1, _⟩ =>
            ⟨w', hw',
              Eq.trans (congrArg w' (Eq.symm hc0)) (Eq.trans e0 (congrArg w hc0)),
              Eq.trans (congrArg w' (Eq.symm hc1)) (Eq.trans e1 (congrArg w hc1))⟩
    | m + 1, _, w, hw, ⟨k + 1, hk⟩ =>
        let q : Fin (m + 1) := ⟨k, Nat.lt_of_succ_lt_succ hk⟩
        have hc1 : q.succ.castSucc = (⟨k + 1, hk⟩ : Fin (m + 2)).castSucc := fin_eq_of_val_eq rfl
        have hc2 : q.succ.succ = (⟨k + 1, hk⟩ : Fin (m + 2)).succ := fin_eq_of_val_eq rfl
        match hw q with
        | ⟨w', hw', _, e1, e2⟩ =>
            ⟨w', hw',
              Eq.trans (congrArg w' (Eq.symm hc1)) (Eq.trans e1 (congrArg w hc1)),
              Eq.trans (congrArg w' (Eq.symm hc2)) (Eq.trans e2 (congrArg w hc2))⟩

/-- Provenance fiber separation theorem: keeping incompatible branch sets S_b in separate
    abstract fibers prevents cross-branch concretization chimeras. -/
theorem provenance_fibers_refine_merged {W A : Type}
    {α : (W → Prop) → A} {γ : A → (W → Prop)} {leA : A → A → Prop}
    (G : GaloisConn α γ leA) {B : Type} (fibers : B → W → Prop) :
    SubsetPred (fun w => ∃ b, γ (α (fibers b)) w)
               (γ (α (fun w => ∃ b, fibers b w))) :=
  fun w ⟨b, hwb⟩ =>
    let hsub : SubsetPred (fibers b) (fun x => ∃ b', fibers b' x) := fun _ hx => ⟨b, hx⟩
    gc_mono_gamma G (gc_mono_alpha G hsub) w hwb

-- 2.5 Repaired Horizon-Support Soundness Theorem for Multiplicative Matching (LBF-16)
structure FixedRewrite (L : Nat) where
  mask : Fin L → Bool
  lhs  : Fin L → Sym
  rhs  : Fin L → Sym

def MatchesFixed {L : Nat} (r : FixedRewrite L) (w : WordF L) : Prop :=
  ∀ p : Fin L, r.mask p = true → w p = r.lhs p

def ApplyFixed {L : Nat} (r : FixedRewrite L) (w : WordF L) : WordF L :=
  fun p => if r.mask p then r.rhs p else w p

def ConcreteStepSet {L : Nat} (R : List (FixedRewrite L)) (S : WordF L → Prop) : WordF L → Prop :=
  fun w' => S w' ∨ ∃ w, S w ∧ ∃ r, r ∈ R ∧ MatchesFixed r w ∧ w' = ApplyFixed r w

theorem concreteStepSet_mono {L : Nat} (R : List (FixedRewrite L))
    {P Q : WordF L → Prop} (hPQ : SubsetPred P Q) :
    SubsetPred (ConcreteStepSet R P) (ConcreteStepSet R Q) :=
  fun w' hw' =>
    match hw' with
    | Or.inl hp => Or.inl (hPQ w' hp)
    | Or.inr ⟨w, hw, r, hr, hm, heq⟩ => Or.inr ⟨w, hPQ w hw, r, hr, hm, heq⟩

/-- Multiplicative window matching abstract transfer on positionwise supports:
    fires rule r whenever every masked slot p has A p (r.lhs p). -/
def MultiplicativePos1Transfer {L : Nat} (R : List (FixedRewrite L)) (A : Abs1 L) : Abs1 L :=
  fun p s =>
    A p s ∨ ∃ r, r ∈ R ∧ (∀ q : Fin L, r.mask q = true → A q (r.lhs q)) ∧ r.mask p = true ∧ r.rhs p = s

theorem multiplicative_pos1_step_sound {L : Nat} (R : List (FixedRewrite L)) (A : Abs1 L) :
    SubsetPred (ConcreteStepSet R (gamma1 A)) (gamma1 (MultiplicativePos1Transfer R A)) :=
  fun _ hw' p =>
    match hw' with
    | Or.inl hA => Or.inl (hA p)
    | Or.inr ⟨w, hwA, r, hr, hmatch, heq⟩ =>
        let goalAt : Sym → Prop := fun s => MultiplicativePos1Transfer R A p s
        have h_applied : goalAt (ApplyFixed r w p) :=
          match hmp : r.mask p with
          | false =>
              have h_eq : ApplyFixed r w p = w p :=
                if_neg (fun ht => Bool.noConfusion (Eq.trans (Eq.symm hmp) ht))
              Eq.subst (motive := goalAt) (Eq.symm h_eq) (Or.inl (hwA p))
          | true =>
              have h_eq : ApplyFixed r w p = r.rhs p := if_pos hmp
              have h_all : ∀ q : Fin L, r.mask q = true → A q (r.lhs q) :=
                fun q hmq =>
                  Eq.subst (motive := fun x => A q x) (hmatch q hmq) (hwA q)
              Eq.subst (motive := goalAt) (Eq.symm h_eq) (Or.inr ⟨r, hr, h_all, hmp, rfl⟩)
        Eq.subst (motive := goalAt) (Eq.symm (congrFun heq p)) h_applied

/-- REPAIRED LBF-16 HORIZON-SUPPORT THEOREM:
    For all horizons k >= 0, concrete <=k-step rewrite reachability from w₀ is a subset of
    the concretization of k-pass multiplicative positionwise support (sound overapproximation). -/
theorem repaired_lbf16_horizon_support_soundness {L : Nat}
    (R : List (FixedRewrite L)) (w₀ : WordF L) (k : Nat) :
    SubsetPred
      (IterConcrete (ConcreteStepSet R) k (fun w => w = w₀))
      (gamma1 (IterAbstract (MultiplicativePos1Transfer R) k (alpha1 (fun w => w = w₀)))) :=
  abstract_iter_soundness
    (ConcreteStepSet R)
    (concreteStepSet_mono R)
    gamma1
    (MultiplicativePos1Transfer R)
    (multiplicative_pos1_step_sound R)
    k
    (fun w => w = w₀)
    (alpha1 (fun w => w = w₀))
    (gc_extensive (galois_pos1 L) (fun w => w = w₀))

-- 2.6 Non-Backtracking Root Decomposition Theorem (Provenance Without Root Re-Mixing)
inductive Walk {W : Type} (step : W → W → Prop) : Nat → W → W → Prop where
  | refl (u : W) : Walk step 0 u u
  | cons {n : Nat} {u v z : W} : step u v → Walk step n v z → Walk step (n + 1) u z

theorem walk_split_last_root {W : Type} [DecidableEq W]
    (step : W → W → Prop) (w₀ : W) {k : Nat} {u z : W} (hw : Walk step k u z) :
    (z = w₀ ∨
      ∃ (w₁ : W) (m : Nat),
        w₁ ≠ w₀ ∧ step w₀ w₁ ∧ m + 1 ≤ k ∧
        Walk (fun x y => step x y ∧ y ≠ w₀) m w₁ z) ∨
    (u ≠ w₀ ∧ Walk (fun x y => step x y ∧ y ≠ w₀) k u z) :=
  @Walk.rec W step
    (fun k u z _ =>
      (z = w₀ ∨
        ∃ (w₁ : W) (m : Nat),
          w₁ ≠ w₀ ∧ step w₀ w₁ ∧ m + 1 ≤ k ∧
          Walk (fun x y => step x y ∧ y ≠ w₀) m w₁ z) ∨
      (u ≠ w₀ ∧ Walk (fun x y => step x y ∧ y ≠ w₀) k u z))
    (fun u =>
      match decEq u w₀ with
      | isTrue hu => Or.inl (Or.inl hu)
      | isFalse hu => Or.inr ⟨hu, Walk.refl u⟩)
    (fun {n u v _} huv _ ih =>
      match ih with
      | Or.inl (Or.inl hz0) => Or.inl (Or.inl hz0)
      | Or.inl (Or.inr ⟨w₁, m, hw1, hs1, hle, hwalk⟩) =>
          Or.inl (Or.inr ⟨w₁, m, hw1, hs1, Nat.le_trans hle (Nat.le_succ n), hwalk⟩)
      | Or.inr ⟨hv_ne, hw_vz⟩ =>
          match decEq u w₀ with
          | isTrue hu =>
              have hs0 : step w₀ v := Eq.subst (motive := fun q => step q v) hu huv
              Or.inl (Or.inr ⟨v, n, hv_ne, hs0, Nat.le_refl (n + 1), hw_vz⟩)
          | isFalse hu =>
              Or.inr ⟨hu, Walk.cons ⟨huv, hv_ne⟩ hw_vz⟩)
    k u z hw

/-- NON-BACKTRACKING PROVENANCE SOUNDNESS THEOREM:
    Every concrete walk of length k from w₀ to z either terminates at z = w₀ (identity fiber)
    or decomposes into an initial branch step w₀ → w₁ (w₁ ≠ w₀) followed by a walk of length
    m <= k - 1 inside w₁'s branch fiber that NEVER revisits w₀! -/
theorem nonbacktracking_provenance_soundness {W : Type} [DecidableEq W]
    (step : W → W → Prop) (w₀ z : W) (k : Nat) (hw : Walk step k w₀ z) :
    z = w₀ ∨
    ∃ (w₁ : W) (m : Nat),
      w₁ ≠ w₀ ∧ step w₀ w₁ ∧ m + 1 ≤ k ∧
      Walk (fun x y => step x y ∧ y ≠ w₀) m w₁ z :=
  match walk_split_last_root step w₀ hw with
  | Or.inl h => h
  | Or.inr ⟨hne, _⟩ => False.elim (hne rfl)


-- ============================================================================
-- Section 3. Executable Abstract Interpreters & Kernel-Checked Counterexamples
-- ============================================================================

abbrev WordL := List Sym

structure SymRule where
  u : WordL
  v : WordL

def orientRules (rs : List SymRule) : List (WordL × WordL) :=
  rs.foldr (fun r acc => (r.u, r.v) :: (r.v, r.u) :: acc) []

def sliceEq : WordL → WordL → Bool
  | [], _ => true
  | _ :: _, [] => false
  | x :: xs, y :: ys => (x == y) && sliceEq xs ys

def replaceAt : Nat → Nat → WordL → WordL → WordL
  | 0, len, rhs, w => rhs ++ w.drop len
  | p + 1, len, rhs, x :: xs => x :: replaceAt p len rhs xs
  | _ + 1, _, _, [] => []

def applyOrientedAt (lhs rhs w : WordL) (p : Nat) : Option WordL :=
  if p + lhs.length ≤ w.length && sliceEq lhs (w.drop p) then
    some (replaceAt p lhs.length rhs w)
  else
    none

def rangeNat : Nat → List Nat
  | 0 => []
  | n + 1 => rangeNat n ++ [n]

def dedupWords : List WordL → List WordL
  | [] => []
  | w :: ws =>
      let rest := dedupWords ws
      if rest.contains w then rest else w :: rest

def stepWord (ors : List (WordL × WordL)) (w : WordL) : List WordL :=
  let positions := rangeNat w.length
  dedupWords <|
    ors.foldr (fun (lhs, rhs) acc =>
      (positions.filterMap (applyOrientedAt lhs rhs w)) ++ acc) [w]

def strictNeighbors (ors : List (WordL × WordL)) (w : WordL) : List WordL :=
  (stepWord ors w).filter (fun w' => w' != w)

def concreteReach (rs : List SymRule) : Nat → List WordL → List WordL
  | 0, ws => dedupWords ws
  | k + 1, ws =>
      let ors := orientRules rs
      let next := dedupWords (ws.foldr (fun w acc => stepWord ors w ++ acc) [])
      concreteReach rs k next

def allSyms : List Sym := [Sym.a, Sym.b, Sym.c, Sym.d, Sym.e, Sym.f]

-- 3.1 Executable Positionwise (1-gram) Abstract Interpreter (LBF-16 Support Semantics)
abbrev Pos1State := List (List Sym)

def dedupSyms : List Sym → List Sym
  | [] => []
  | s :: ss =>
      let r := dedupSyms ss
      if r.contains s then r else s :: r

def initPos1 (w : WordL) : Pos1State :=
  w.map (fun s => [s])

def getPos1 (st : Pos1State) (p : Nat) : List Sym :=
  match st.drop p with
  | col :: _ => col
  | [] => []

def addSymAt (st : Pos1State) (p : Nat) (s : Sym) : Pos1State :=
  (rangeNat st.length).map (fun i =>
    let col := getPos1 st i
    if i == p then dedupSyms (s :: col) else col)

def windowMatchesPos1 (st : Pos1State) (p : Nat) : WordL → Bool
  | [] => true
  | x :: xs => (getPos1 st p).contains x && windowMatchesPos1 st (p + 1) xs

def writeRhsPos1 (st : Pos1State) (p : Nat) : WordL → Pos1State
  | [] => st
  | y :: ys => writeRhsPos1 (addSymAt st p y) (p + 1) ys

def stepPos1 (ors : List (WordL × WordL)) (L : Nat) (st : Pos1State) : Pos1State :=
  ors.foldl (fun acc (lhs, rhs) =>
    (rangeNat L).foldl (fun acc2 p =>
      if p + lhs.length ≤ L && windowMatchesPos1 st p lhs then
        writeRhsPos1 acc2 p rhs
      else
        acc2) acc) st

def runPos1 (rs : List SymRule) (k : Nat) (w : WordL) : Pos1State :=
  let ors := orientRules rs
  let rec loop : Nat → Pos1State → Pos1State
    | 0, st => st
    | n + 1, st => loop n (stepPos1 ors w.length st)
  loop k (initPos1 w)

-- 3.2 Executable Trellis-Consistent n-Gram Abstract Interpreter (n = 2 and n = 3)
def extractNGrams (n : Nat) (w : WordL) : List (Nat × WordL) :=
  (rangeNat (w.length + 1 - n)).map (fun p => (p, (w.drop p).take n))

def alphaNGram (n : Nat) (ws : List WordL) : List (Nat × WordL) :=
  ws.foldr (fun w acc =>
    (extractNGrams n w).foldr (fun g a2 => if a2.contains g then a2 else g :: a2) acc) []

def extendTrellis (n : Nat) (table : List (Nat × WordL)) (p : Nat) (prefixes : List WordL) : List WordL :=
  prefixes.foldr (fun pref acc =>
    let tail := pref.drop (pref.length + 1 - n)
    let exts := allSyms.filterMap (fun s =>
      let gram := tail ++ [s]
      if table.contains (p, gram) then some (pref ++ [s]) else none)
    exts ++ acc) []

def gammaNGram (n : Nat) (L : Nat) (table : List (Nat × WordL)) : List WordL :=
  let seeds := (table.filter (fun (p, _) => p == 0)).map Prod.snd
  let grown := (rangeNat (L - n)).foldl (fun cur idx => extendTrellis n table (idx + 1) cur) seeds
  dedupWords grown

def runNGram (n : Nat) (rs : List SymRule) (k : Nat) (w : WordL) : List (Nat × WordL) :=
  let L := w.length
  let ors := orientRules rs
  let rec loop : Nat → List (Nat × WordL) → List (Nat × WordL)
    | 0, tbl => tbl
    | m + 1, tbl =>
        let concrete := gammaNGram n L tbl
        let stepped := dedupWords (concrete.foldr (fun u acc => stepWord ors u ++ acc) [])
        loop m (alphaNGram n stepped)
  loop k (alphaNGram n [w])

-- 3.3 Executable Branch-Provenance Fibered Abstract Interpreters:
-- (A) Backtracking 1-step branch provenance + Pos1 (delays chimera from pass 2 to pass 4!)
def runBacktrackingProvPos1 (rs : List SymRule) (k : Nat) (w₀ : WordL) : List Pos1State :=
  match k with
  | 0 => [initPos1 w₀]
  | m + 1 =>
      let ors := orientRules rs
      initPos1 w₀ :: (strictNeighbors ors w₀).map (fun w₁ => runPos1 rs m w₁)

-- (B) Non-backtracking branch-provenance fibers with Pos1 (and with Adj2!)
def stepPos1NonBacktrack (ors : List (WordL × WordL)) (w₀ : WordL) (st : Pos1State) : Pos1State :=
  let L := w₀.length
  ors.foldl (fun acc (lhs, rhs) =>
    (rangeNat L).foldl (fun acc2 p =>
      if p + lhs.length ≤ L && windowMatchesPos1 st p lhs then
        if sliceEq rhs (w₀.drop p) then acc2
        else writeRhsPos1 acc2 p rhs
      else
        acc2) acc) st

def runNonBacktrackProvPos1 (rs : List SymRule) (k : Nat) (w₀ : WordL) : List Pos1State :=
  match k with
  | 0 => [initPos1 w₀]
  | m + 1 =>
      let ors := orientRules rs
      let rec loop : Nat → Pos1State → Pos1State
        | 0, st => st
        | r + 1, st => loop r (stepPos1NonBacktrack ors w₀ st)
      initPos1 w₀ :: (strictNeighbors ors w₀).map (fun w₁ => loop m (initPos1 w₁))

def runNonBacktrackProvAdj2Words (rs : List SymRule) (k : Nat) (w₀ : WordL) : List WordL :=
  let L := w₀.length
  let ors := orientRules rs
  match k with
  | 0 => [w₀]
  | m + 1 =>
      let stepNoRoot (u : WordL) : List WordL :=
        (stepWord ors u).filter (fun z => z != w₀)
      let rec loop : Nat → List (Nat × WordL) → List (Nat × WordL)
        | 0, tbl => tbl
        | r + 1, tbl =>
            let concrete := gammaNGram 2 L tbl
            let stepped := dedupWords (concrete.foldr (fun u acc => stepNoRoot u ++ acc) [])
            loop r (alphaNGram 2 stepped)
      let fiberTables := (strictNeighbors ors w₀).map (fun w₁ => loop m (alphaNGram 2 [w₁]))
      dedupWords (w₀ :: fiberTables.foldr (fun tbl acc => gammaNGram 2 L tbl ++ acc) [])

def provHasSymAt (fibers : List Pos1State) (p : Nat) (s : Sym) : Bool :=
  fibers.any (fun st => (getPos1 st p).contains s)


-- ============================================================================
-- Counterexample 1 (Mandatory LBF-16 Regression):
--   w₀ = "abc", Rules: ab ↔ de, bc ↔ fa, df ↔ cc
--   Concrete reachable component: {abc, dec, afa}
-- ============================================================================

def w_cex1 : WordL := [Sym.a, Sym.b, Sym.c]
def rules_cex1 : List SymRule := [
  ⟨[Sym.a, Sym.b], [Sym.d, Sym.e]⟩,
  ⟨[Sym.b, Sym.c], [Sym.f, Sym.a]⟩,
  ⟨[Sym.d, Sym.f], [Sym.c, Sym.c]⟩
]

theorem cex1_concrete_component_exact :
    (concreteReach rules_cex1 4 [w_cex1]).length = 3 ∧
    (concreteReach rules_cex1 4 [w_cex1]).contains [Sym.a, Sym.b, Sym.c] = true ∧
    (concreteReach rules_cex1 4 [w_cex1]).contains [Sym.d, Sym.e, Sym.c] = true ∧
    (concreteReach rules_cex1 4 [w_cex1]).contains [Sym.a, Sym.f, Sym.a] = true ∧
    (concreteReach rules_cex1 4 [w_cex1]).any (fun w => w.take 2 == [Sym.c, Sym.c]) = false :=
  ⟨rfl, rfl, rfl, rfl, rfl⟩

theorem cex1_pos1_pass1_exact_pass2_spurious :
    (getPos1 (runPos1 rules_cex1 1 w_cex1) 0).contains Sym.c = false ∧
    (getPos1 (runPos1 rules_cex1 1 w_cex1) 1).contains Sym.c = false ∧
    (getPos1 (runPos1 rules_cex1 1 w_cex1) 0).contains Sym.d = true ∧
    (getPos1 (runPos1 rules_cex1 1 w_cex1) 1).contains Sym.f = true ∧
    (getPos1 (runPos1 rules_cex1 2 w_cex1) 0).contains Sym.c = true ∧
    (getPos1 (runPos1 rules_cex1 2 w_cex1) 1).contains Sym.c = true :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem cex1_relational_and_provenance_exact :
    -- Backtracking provenance is exact through Pass 3, but re-mixes via w₁→w₀→w₂ at Pass 4:
    provHasSymAt (runBacktrackingProvPos1 rules_cex1 3 w_cex1) 0 Sym.c = false ∧
    provHasSymAt (runBacktrackingProvPos1 rules_cex1 3 w_cex1) 1 Sym.c = false ∧
    provHasSymAt (runBacktrackingProvPos1 rules_cex1 4 w_cex1) 0 Sym.c = true ∧
    -- 2-gram, 3-gram, and non-backtracking provenance remain exact at Pass 4 and beyond:
    (runNGram 2 rules_cex1 4 w_cex1).contains (0, [Sym.d, Sym.f]) = false ∧
    (runNGram 2 rules_cex1 4 w_cex1).contains (0, [Sym.c, Sym.c]) = false ∧
    (gammaNGram 2 3 (runNGram 2 rules_cex1 4 w_cex1)).length = 3 ∧
    (gammaNGram 3 3 (runNGram 3 rules_cex1 4 w_cex1)).length = 3 ∧
    provHasSymAt (runNonBacktrackProvPos1 rules_cex1 4 w_cex1) 0 Sym.c = false ∧
    provHasSymAt (runNonBacktrackProvPos1 rules_cex1 4 w_cex1) 1 Sym.c = false :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩


-- ============================================================================
-- Counterexample 2 (2-Gram Shared-Middle Failure vs 3-Gram & Provenance):
--   w₀ = "abbc" (L = 4), Rules: ab ↔ de, bbc ↔ efa, def ↔ ccc
--   Concrete reachable component: {abbc, debc, aefa} (size 3)
-- ============================================================================

def w_cex2 : WordL := [Sym.a, Sym.b, Sym.b, Sym.c]
def rules_cex2 : List SymRule := [
  ⟨[Sym.a, Sym.b], [Sym.d, Sym.e]⟩,
  ⟨[Sym.b, Sym.b, Sym.c], [Sym.e, Sym.f, Sym.a]⟩,
  ⟨[Sym.d, Sym.e, Sym.f], [Sym.c, Sym.c, Sym.c]⟩
]

theorem cex2_adj2_spurious_sub3_and_prov_exact :
    (concreteReach rules_cex2 3 [w_cex2]).length = 3 ∧
    (concreteReach rules_cex2 3 [w_cex2]).any (fun w => w.take 3 == [Sym.c, Sym.c, Sym.c]) = false ∧
    (gammaNGram 2 4 (runNGram 2 rules_cex2 1 w_cex2)).contains [Sym.d, Sym.e, Sym.f, Sym.a] = true ∧
    (runNGram 2 rules_cex2 2 w_cex2).contains (0, [Sym.c, Sym.c]) = true ∧
    (gammaNGram 3 4 (runNGram 3 rules_cex2 3 w_cex2)).length = 3 ∧
    (runNGram 3 rules_cex2 3 w_cex2).contains (0, [Sym.c, Sym.c, Sym.c]) = false ∧
    provHasSymAt (runNonBacktrackProvPos1 rules_cex2 3 w_cex2) 0 Sym.c = false :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩


-- ============================================================================
-- Counterexample 3 (Local 3-Gram Loses Global Consistency vs Provenance Fibers):
--   w₀ = "aebcd" (L = 5), Rules: aeb ↔ def, bcd ↔ ffa, ff ↔ cc, dec ↔ bbb
--   Concrete component: {aebcd, defcd, aeffa, aecca} (size 4; "bbb" never occurs!)
-- ============================================================================

def w_cex3 : WordL := [Sym.a, Sym.e, Sym.b, Sym.c, Sym.d]
def rules_cex3 : List SymRule := [
  ⟨[Sym.a, Sym.e, Sym.b], [Sym.d, Sym.e, Sym.f]⟩,
  ⟨[Sym.b, Sym.c, Sym.d], [Sym.f, Sym.f, Sym.a]⟩,
  ⟨[Sym.f, Sym.f], [Sym.c, Sym.c]⟩,
  ⟨[Sym.d, Sym.e, Sym.c], [Sym.b, Sym.b, Sym.b]⟩
]

theorem cex3_sub3_loses_global_consistency_prov_exact :
    (concreteReach rules_cex3 4 [w_cex3]).length = 4 ∧
    (concreteReach rules_cex3 4 [w_cex3]).contains [Sym.d, Sym.e, Sym.f, Sym.f, Sym.a] = false ∧
    (concreteReach rules_cex3 4 [w_cex3]).any (fun w => w.take 3 == [Sym.b, Sym.b, Sym.b]) = false ∧
    (gammaNGram 3 5 (runNGram 3 rules_cex3 1 w_cex3)).contains [Sym.d, Sym.e, Sym.f, Sym.f, Sym.a] = true ∧
    (runNGram 3 rules_cex3 2 w_cex3).contains (0, [Sym.d, Sym.e, Sym.c]) = true ∧
    (runNGram 3 rules_cex3 3 w_cex3).contains (0, [Sym.b, Sym.b, Sym.b]) = true ∧
    (runNonBacktrackProvAdj2Words rules_cex3 4 w_cex3).length = 4 ∧
    (runNonBacktrackProvAdj2Words rules_cex3 4 w_cex3).contains
      [Sym.d, Sym.e, Sym.f, Sym.f, Sym.a] = false ∧
    provHasSymAt (runNonBacktrackProvPos1 rules_cex3 4 w_cex3) 0 Sym.b = false :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

end LBF17AbstractInterp

#print axioms LBF17AbstractInterp.abstract_iter_soundness
#print axioms LBF17AbstractInterp.best_transformer_soundness
#print axioms LBF17AbstractInterp.galois_pos1
#print axioms LBF17AbstractInterp.galois_adj2
#print axioms LBF17AbstractInterp.galois_sub3
#print axioms LBF17AbstractInterp.adj2_refines_pos1
#print axioms LBF17AbstractInterp.sub3_refines_adj2
#print axioms LBF17AbstractInterp.provenance_fibers_refine_merged
#print axioms LBF17AbstractInterp.repaired_lbf16_horizon_support_soundness
#print axioms LBF17AbstractInterp.nonbacktracking_provenance_soundness
#print axioms LBF17AbstractInterp.cex1_concrete_component_exact
#print axioms LBF17AbstractInterp.cex1_pos1_pass1_exact_pass2_spurious
#print axioms LBF17AbstractInterp.cex1_relational_and_provenance_exact
#print axioms LBF17AbstractInterp.cex2_adj2_spurious_sub3_and_prov_exact
#print axioms LBF17AbstractInterp.cex3_sub3_loses_global_consistency_prov_exact
