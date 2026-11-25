/-
  Independent referee audit appendix for LBF17AbstractInterp.lean.

  This file is NOT standalone: `audit_02_lean_recheck.py` appends it to an unmodified copy
  of the registered `lean/LBF17AbstractInterp.lean` (SHA-256 1d34145d...) in a scratch
  directory and compiles the concatenation with Lean 4.19.0.  Every statement below is about
  the registered definitions exactly as written; all proofs are kernel `rfl`.

  Instance R* (all rule sides distinct, lengths 2 and 3, as in the LSA-02K generator):
        abc <-> def        ab <-> de        bf <-> cc           w0 = abc
  Concrete walk:  abc -> def -> abf -> acc.
-/

namespace LBF17Audit
open LBF17AbstractInterp

def w_star : WordL := [Sym.a, Sym.b, Sym.c]

def rules_star : List SymRule := [
  ⟨[Sym.a, Sym.b, Sym.c], [Sym.d, Sym.e, Sym.f]⟩,
  ⟨[Sym.a, Sym.b], [Sym.d, Sym.e]⟩,
  ⟨[Sym.b, Sym.f], [Sym.c, Sym.c]⟩
]

/-- Membership of a word in the Cartesian concretisation of one positionwise fiber. -/
def inGammaPos1 (st : Pos1State) (w : WordL) : Bool :=
  (rangeNat w.length).all (fun p =>
    match w.drop p with
    | s :: _ => (getPos1 st p).contains s
    | [] => true)

/-- The concrete component: five words, with `acc` at distance exactly 3. -/
theorem audit_concrete_component :
    (concreteReach rules_star 6 [w_star]).length = 5 ∧
    (concreteReach rules_star 2 [w_star]).contains [Sym.a, Sym.b, Sym.f] = true ∧
    (concreteReach rules_star 2 [w_star]).contains [Sym.a, Sym.c, Sym.c] = false ∧
    (concreteReach rules_star 3 [w_star]).contains [Sym.a, Sym.c, Sym.c] = true :=
  ⟨rfl, rfl, rfl, rfl⟩

/-- WORD-LEVEL UNSOUNDNESS of the registered executable non-backtracking provenance
    interpreter: the 2-step reachable word `abf` lies in no fiber concretisation, at pass 2
    or at any later pass checked (the slice-equality suppression never lifts). -/
theorem audit_nonbacktrack_prov_pos1_loses_reachable_word :
    (runNonBacktrackProvPos1 rules_star 2 w_star).any (fun st => inGammaPos1 st [Sym.a, Sym.b, Sym.f]) = false ∧
    (runNonBacktrackProvPos1 rules_star 4 w_star).any (fun st => inGammaPos1 st [Sym.a, Sym.b, Sym.f]) = false ∧
    (runNonBacktrackProvPos1 rules_star 8 w_star).any (fun st => inGammaPos1 st [Sym.a, Sym.b, Sym.f]) = false :=
  ⟨rfl, rfl, rfl⟩

/-- SYMBOL-LEVEL UNSOUNDNESS: symbol `c` is concretely reachable at position 1 within three
    steps, yet no fiber of `runNonBacktrackProvPos1` ever supports it (passes 3 through 8).
    Hence "zero bank certifies unreachability" fails for this interpreter. -/
theorem audit_nonbacktrack_prov_pos1_symbol_false_negative :
    (concreteReach rules_star 3 [w_star]).any (fun w => (w.drop 1).take 1 == [Sym.c]) = true ∧
    provHasSymAt (runNonBacktrackProvPos1 rules_star 3 w_star) 1 Sym.c = false ∧
    provHasSymAt (runNonBacktrackProvPos1 rules_star 4 w_star) 1 Sym.c = false ∧
    provHasSymAt (runNonBacktrackProvPos1 rules_star 8 w_star) 1 Sym.c = false :=
  ⟨rfl, rfl, rfl, rfl⟩

/-- The plain (non-provenance) positionwise and bigram interpreters are sound here, so the
    defect is introduced by the suppression rule, not by the base domains. -/
theorem audit_base_domains_sound_on_instance :
    (getPos1 (runPos1 rules_star 3 w_star) 1).contains Sym.c = true ∧
    (gammaNGram 2 3 (runNGram 2 rules_star 3 w_star)).contains [Sym.a, Sym.c, Sym.c] = true :=
  ⟨rfl, rfl⟩

/-- The file's OTHER non-backtracking model, which filters successor WORDS equal to w0
    (the semantics that `nonbacktracking_provenance_soundness` actually justifies), is sound
    on the same instance.  The two registered executable models therefore disagree, and only
    the unsound one mirrors the slice-based suppression used by the PyTorch cell. -/
theorem audit_word_filter_model_is_sound_on_instance :
    (runNonBacktrackProvAdj2Words rules_star 3 w_star).contains [Sym.a, Sym.c, Sym.c] = true ∧
    (runNonBacktrackProvAdj2Words rules_star 2 w_star).contains [Sym.a, Sym.b, Sym.f] = true :=
  ⟨rfl, rfl⟩

end LBF17Audit

#print axioms LBF17Audit.audit_concrete_component
#print axioms LBF17Audit.audit_nonbacktrack_prov_pos1_loses_reachable_word
#print axioms LBF17Audit.audit_nonbacktrack_prov_pos1_symbol_false_negative
#print axioms LBF17Audit.audit_base_domains_sound_on_instance
#print axioms LBF17Audit.audit_word_filter_model_is_sound_on_instance
