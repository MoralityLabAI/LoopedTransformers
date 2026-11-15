/-!
AF-15 abstract host Horn gate. A candidate atom may enter the state only if the
host has independently established its Horn-premise certificate. These pure
Boolean lemmas express pointwise soundness and inflationarity. They do not
certify a neural ranking policy or a proof-search success rate.
-/

theorem af15_sound_union (old fresh reachable : Bool)
    (h_old : old = true → reachable = true)
    (h_fresh : fresh = true → reachable = true)
    (h_new : (old || fresh) = true) : reachable = true := by
  cases old <;> cases fresh <;> cases reachable <;> simp_all

theorem af15_fact_persists (old fired : Bool) (h : old = true) :
    (old || fired) = true := by
  simp [h]

theorem af15_no_unsupported_fire (known enabled closed : Bool)
    (h_known : known = true → closed = true)
    (h_rule : enabled = true → closed = true) :
    (known || enabled) = true → closed = true := by
  intro h
  exact af15_sound_union known enabled closed h_known h_rule h
