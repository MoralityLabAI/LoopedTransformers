/-!
AF-16 host-owned Horn transitions. These Boolean laws apply when the host
checks each rule's actual public premises and consequent. The neural ranker
supplies an action preference and has no authority to add unsupported atoms.
The abstract theorems do not certify neural predictions or held-out success.
-/

theorem af16_host_inflationary (old fired : Bool) (h : old = true) :
    (old || fired) = true := by simp [h]

theorem af16_host_sound (old fired reachable : Bool)
    (hOld : old = true → reachable = true)
    (hFire : fired = true → reachable = true)
    (hNew : (old || fired) = true) : reachable = true := by
  cases old <;> cases fired <;> cases reachable <;> simp_all

theorem af16_host_no_false_goal (old new goal : Bool)
    (hOld : old = true → goal = true)
    (hNew : new = true → goal = true) :
    (old || new) = true → goal = true := by
  intro h
  exact af16_host_sound old new goal hOld hNew h
