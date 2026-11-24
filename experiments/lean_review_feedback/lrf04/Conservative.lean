import MathCan.ConstructiveObservationAudit

namespace LRF04

open MathCan.ConstructiveObservationAudit

/-- Require an inhabited fiber and unanimous permission in that fiber. -/
def conservative {S O A : Type} [DecidableEq O]
    (obs : S → O) (auth : S → A → Bool) (xs : List S) (o : O) (a : A) : Bool :=
  xs.any (fun s => decide (obs s = o)) &&
    xs.all (fun s => if obs s = o then auth s a else true)

theorem conservative_sound {S O A : Type} [DecidableEq O]
    (obs : S → O) (auth : S → A → Bool) (xs : List S) (o : O) (a : A)
    (s : S) (hm : s ∈ xs) (ho : obs s = o)
    (h : conservative obs auth xs o a = true) : auth s a = true := by
  have hp : (xs.any (fun t => decide (obs t = o)) &&
      xs.all (fun t => if obs t = o then auth t a else true)) = true := h
  have hall := (Bool.and_eq_true_iff.mp hp).2
  have hs := (List.all_eq_true.mp hall) s hm
  simpa [ho] using hs

def states : List (Bool × Bool) :=
  [(false,false),(false,true),(true,false),(true,true)]

def collisionAuth (s : Bool × Bool) (a : Bool) : Bool := s.1 && s.2 && a

def compatible (o : Nat) (a : Bool) : Bool :=
  conservative demoObserve demoAuthorize states o a

def collision (o : Nat) (a : Bool) : Bool :=
  conservative demoObserve collisionAuth states o a

theorem states_complete (s : Bool × Bool) : s ∈ states := by
  rcases s with ⟨x,y⟩
  cases x <;> cases y <;> decide

theorem compatible_matches_demo (o : Nat) (a : Bool) :
    compatible o a = demoMonitor o a := by
  cases o with
  | zero => cases a <;> rfl
  | succ n =>
    cases n with
    | zero => cases a <;> rfl
    | succ n => cases a <;> rfl

theorem collision_denied (a : Bool) : collision 1 a = false := by
  cases a <;> rfl

theorem unknown_denied (a : Bool) :
    compatible 2 a = false ∧ collision 2 a = false := by
  cases a <;> decide

#print axioms conservative_sound
#print axioms states_complete
#print axioms compatible_matches_demo
#print axioms collision_denied
#print axioms unknown_denied

#eval IO.println ("LRF04_NATIVE " ++ reprStr
    ([compatible, collision].flatMap fun f =>
      [0,1,2].flatMap fun o => [false,true].map fun a => f o a))

end LRF04
