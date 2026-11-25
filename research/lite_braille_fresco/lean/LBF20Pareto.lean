/-
LBF-20C. Algebra of premise sets and sequential rewrite lengths.
Predicates avoid propositional extensionality: equivalence is mutual domination.
-/
namespace LBF20Pareto

structure Label where
  guards : Nat → Prop
  work : Nat

def Dom (a b : Label) : Prop :=
  (∀ g, a.guards g → b.guards g) ∧ a.work ≤ b.work

def mul (a b : Label) : Label :=
  ⟨fun g => a.guards g ∨ b.guards g, a.work + b.work⟩

def one : Label := ⟨fun _ => False, 0⟩

def single (g : Nat) : Label := ⟨fun h => h = g, 1⟩

def Feasible (a : Label) (allowed : Nat → Prop) (H : Nat) : Prop :=
  (∀ g, a.guards g → allowed g) ∧ a.work ≤ H

theorem dom_refl (a : Label) : Dom a a := ⟨fun _ h => h, Nat.le_refl _⟩

theorem dom_trans {a b c : Label} (h : Dom a b) (k : Dom b c) : Dom a c :=
  ⟨fun g x => k.1 g (h.1 g x), Nat.le_trans h.2 k.2⟩

theorem mul_mono {a b c d : Label} (h : Dom a b) (k : Dom c d) :
    Dom (mul a c) (mul b d) :=
  ⟨fun g x => x.elim (fun t => Or.inl (h.1 g t)) (fun t => Or.inr (k.1 g t)),
    Nat.add_le_add h.2 k.2⟩

theorem assoc_forward (a b c : Label) : Dom (mul (mul a b) c) (mul a (mul b c)) :=
  ⟨fun _ h => h.elim (fun x => x.elim Or.inl (fun y => Or.inr (Or.inl y)))
    (fun z => Or.inr (Or.inr z)),
    (Nat.add_assoc a.work b.work c.work) ▸ Nat.le_refl ((a.work + b.work) + c.work)⟩

theorem assoc_backward (a b c : Label) : Dom (mul a (mul b c)) (mul (mul a b) c) :=
  ⟨fun _ h => h.elim (fun x => Or.inl (Or.inl x))
    (fun y => y.elim (fun z => Or.inl (Or.inr z)) Or.inr),
    (Nat.add_assoc a.work b.work c.work).symm ▸ Nat.le_refl (a.work + (b.work + c.work))⟩

theorem one_left (a : Label) : Dom (mul one a) a :=
  ⟨fun _ h => h.elim False.elim (fun x => x), (Nat.zero_add a.work).symm ▸ Nat.le_refl a.work⟩

theorem one_left_reverse (a : Label) : Dom a (mul one a) :=
  ⟨fun _ h => Or.inr h, (Nat.zero_add a.work).symm ▸ Nat.le_refl a.work⟩

theorem reuse_guard (g h : Nat) :
    (mul (single g) (single g)).guards h ↔ (single g).guards h :=
  ⟨fun x => x.elim (fun y => y) (fun y => y), fun x => Or.inl x⟩

theorem feasible_down {a b : Label} {allowed : Nat → Prop} {H : Nat}
    (h : Dom a b) (k : Feasible b allowed H) : Feasible a allowed H :=
  ⟨fun g x => k.1 g (h.1 g x), Nat.le_trans h.2 k.2⟩

abbrev Frontier := Label → Prop

def Covers (F G : Frontier) : Prop := ∀ b, G b → ∃ a, F a ∧ Dom a b

def Product (F G : Frontier) : Frontier :=
  fun c => ∃ a b, F a ∧ G b ∧ c = mul a b

def Query (F : Frontier) (allowed : Nat → Prop) (H : Nat) : Prop :=
  ∃ a, F a ∧ Feasible a allowed H

theorem covers_trans {F G J : Frontier} (h : Covers F G) (k : Covers G J) :
    Covers F J := by
  intro b hb
  obtain ⟨a, ha, hd⟩ := k b hb
  obtain ⟨c, hc, he⟩ := h a ha
  exact ⟨c, hc, dom_trans he hd⟩

theorem covers_product {F G F' G' : Frontier} (h : Covers F F') (k : Covers G G') :
    Covers (Product F G) (Product F' G') := by
  intro c hc
  obtain ⟨a, b, ha, hb, eq⟩ := hc
  obtain ⟨a', ha', da⟩ := h a ha
  obtain ⟨b', hb', db⟩ := k b hb
  exact ⟨mul a' b', ⟨a', b', ha', hb', rfl⟩, eq.symm ▸ mul_mono da db⟩

theorem covers_union {F G F' G' : Frontier} (h : Covers F F') (k : Covers G G') :
    Covers (fun a => F a ∨ G a) (fun a => F' a ∨ G' a) := by
  intro b hb
  cases hb with
  | inl hl =>
    obtain ⟨a, ha, d⟩ := h b hl
    exact ⟨a, Or.inl ha, d⟩
  | inr hr =>
    obtain ⟨a, ha, d⟩ := k b hr
    exact ⟨a, Or.inr ha, d⟩

theorem query_preserved {F G : Frontier} {allowed : Nat → Prop} {H : Nat}
    (h : Covers F G) (k : Covers G F) :
    Query F allowed H ↔ Query G allowed H := by
  constructor
  · intro ⟨a, ha, fa⟩
    obtain ⟨b, hb, d⟩ := k a ha
    exact ⟨b, hb, feasible_down d fa⟩
  · intro ⟨b, hb, fb⟩
    obtain ⟨a, ha, d⟩ := h b hb
    exact ⟨a, ha, feasible_down d fb⟩

theorem context_prune_safe {a b : Label} (h : Dom a b) (l r : Label)
    {allowed : Nat → Prop} {H : Nat} :
    Feasible (mul (mul l b) r) allowed H →
    Feasible (mul (mul l a) r) allowed H :=
  feasible_down (mul_mono (mul_mono (dom_refl l) h) (dom_refl r))

end LBF20Pareto
#print axioms LBF20Pareto.dom_refl
#print axioms LBF20Pareto.dom_trans
#print axioms LBF20Pareto.mul_mono
#print axioms LBF20Pareto.assoc_forward
#print axioms LBF20Pareto.assoc_backward
#print axioms LBF20Pareto.one_left
#print axioms LBF20Pareto.one_left_reverse
#print axioms LBF20Pareto.reuse_guard
#print axioms LBF20Pareto.feasible_down
#print axioms LBF20Pareto.covers_trans
#print axioms LBF20Pareto.covers_product
#print axioms LBF20Pareto.covers_union
#print axioms LBF20Pareto.query_preserved
#print axioms LBF20Pareto.context_prune_safe
