/- AF-12: elementary monotonicity of the Horn primitives.
   This is a kernel-checked finite Boolean instance, separate from the neural
   model's status and from the general finite-set closure proof in README.md. -/

theorem af12_and_monotone :
  ∀ a a' b b' : Bool,
  ((!a || a') && (!b || b')) = true →
  (!(a && b) || (a' && b')) = true := by decide

theorem af12_or_monotone :
  ∀ a a' b b' : Bool,
  ((!a || a') && (!b || b')) = true →
  (!(a || b) || (a' || b')) = true := by decide

theorem af12_and_inflationary :
  ∀ a b : Bool, (a || (a && b)) = a := by decide

theorem af12_or_inflationary :
  ∀ a b : Bool, (a || (a || b)) = (a || b) := by decide
