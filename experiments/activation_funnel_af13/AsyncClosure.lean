/-!
AF-13: elementary properties of asynchronously applied Boolean Horn rules.
The general finite-fairness result is proved mathematically in README.md;
these are kernel-checkable finite Boolean lemmas, not a Lean proof of the
complete stochastic training algorithm or student theorem predictions.
-/

theorem af13_partial_rule_inflationary :
  ∀ old enabled : Bool, (old || (enabled && old)) = old := by decide

theorem af13_apply_rule_never_erases :
  ∀ old enabled : Bool, (!old || (old || enabled)) = true := by decide

theorem af13_and_rule_isotone :
  ∀ a a' b b' : Bool,
  ((!a || a') && (!b || b')) = true →
  (!(a && b) || (a' && b')) = true := by decide

theorem af13_or_rule_isotone :
  ∀ a a' b b' : Bool,
  ((!a || a') && (!b || b')) = true →
  (!(a || b) || (a' || b')) = true := by decide

theorem af13_fired_head_from_enabled_premises :
  ∀ a b fire : Bool,
  (fire && (a && b)) = true → (a && b) = true := by decide

theorem af13_idempotent_addition :
  ∀ old head : Bool, ((old || head) || head) = (old || head) := by decide
