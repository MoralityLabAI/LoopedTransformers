import Lean
import MathCan.Settlement

/-!
# CHTR-06: an abstract integer flash-accounting kernel

This module establishes conservation for a *declared* abstract transition
system. A proof that a particular deployed EVM hook refines these operations
requires separate bytecode/call-frame/balance evidence. The Python trace
verifier records the corresponding integer transition receipts.

`FlashBook` is the aggregate of one currency across all accounts and pools.
The invariant is

  manager vault = virtual pool amounts + redeemable ERC6909-like claims
                + signed transient caller deltas + unallocated surplus.

Each operation preserves that equality. Two currencies are simply two copies.
-/

namespace MathCan.FlashAccounting

structure FlashBook where
  vault : Int
  pools : Int
  claims : Int
  deltas : Int
  surplus : Int
  deriving Repr, DecidableEq

/-- Aggregate backing equation for one currency. -/
def Backed (s : FlashBook) : Prop :=
  s.vault = s.pools + s.claims + s.deltas + s.surplus

/-- A pool allocation changes the caller's transient obligation oppositely. -/
def poolDelta (s : FlashBook) (amount : Int) : FlashBook :=
  { s with pools := s.pools + amount, deltas := s.deltas - amount }

/-- Direct ERC20 transfer into manager custody enters unallocated surplus. -/
def deposit (s : FlashBook) (amount : Int) : FlashBook :=
  { s with vault := s.vault + amount, surplus := s.surplus + amount }

/-- Direct transfer out of manager custody debits unallocated surplus. -/
def rawWithdrawal (s : FlashBook) (amount : Int) : FlashBook :=
  { s with vault := s.vault - amount, surplus := s.surplus - amount }

/-- `settle` moves deposited surplus into a caller's positive delta. -/
def settle (s : FlashBook) (amount : Int) : FlashBook :=
  { s with deltas := s.deltas + amount, surplus := s.surplus - amount }

/-- `take` moves custody to a recipient, decreasing the caller's delta. -/
def take (s : FlashBook) (amount : Int) : FlashBook :=
  { s with vault := s.vault - amount, deltas := s.deltas - amount }

/-- ERC6909-like mint changes the outstanding liability and caller delta. -/
def mintClaim (s : FlashBook) (amount : Int) : FlashBook :=
  { s with claims := s.claims + amount, deltas := s.deltas - amount }

/-- ERC6909-like burn extinguishes a liability and credits the caller. -/
def burnClaim (s : FlashBook) (amount : Int) : FlashBook :=
  { s with claims := s.claims - amount, deltas := s.deltas + amount }

/-- Exact positive delta forfeiture transfers the benefit to surplus. -/
def clear (s : FlashBook) (amount : Int) : FlashBook :=
  { s with deltas := s.deltas - amount, surplus := s.surplus + amount }

/-- Frame rollback restores the checkpoint, including transient deltas. -/
def revertFrame (checkpoint _attempted : FlashBook) : FlashBook := checkpoint

/-- A conservative abstract opcode vocabulary. -/
inductive Op where
  | pool (amount : Int)
  | deposit (amount : Int)
  | rawWithdrawal (amount : Int)
  | settle (amount : Int)
  | take (amount : Int)
  | mintClaim (amount : Int)
  | burnClaim (amount : Int)
  | clear (amount : Int)
  deriving Repr, DecidableEq

def apply (op : Op) (s : FlashBook) : FlashBook :=
  match op with
  | .pool n => poolDelta s n
  | .deposit n => deposit s n
  | .rawWithdrawal n => rawWithdrawal s n
  | .settle n => settle s n
  | .take n => take s n
  | .mintClaim n => mintClaim s n
  | .burnClaim n => burnClaim s n
  | .clear n => clear s n

/-- Every declared opcode preserves the aggregate backing identity. -/
theorem backed_apply (op : Op) (s : FlashBook) (h : Backed s) :
    Backed (apply op s) := by
  cases op <;> simp [apply, Backed, poolDelta, deposit,
    rawWithdrawal, settle, take, mintClaim, burnClaim, clear] at * <;> omega

/-- Left-to-right execution of a finite abstract trace. -/
def run (s : FlashBook) : List Op → FlashBook
  | [] => s
  | op :: rest => run (apply op s) rest

/-- Conservation holds along arbitrary finite action words. -/
theorem backed_run (s : FlashBook) (h : Backed s) (actions : List Op) :
    Backed (run s actions) := by
  induction actions generalizing s with
  | nil => simpa [run] using h
  | cons op rest ih =>
      simpa [run] using ih (apply op s) (backed_apply op s h)
set_option maxHeartbeats 100000

/-- At lock, zero transient deltas and nonnegative surplus imply vault backing
    for the pool plus outstanding redeemable claims. -/
theorem terminal_vault_solvency (s : FlashBook)
    (h : Backed s) (settled : s.deltas = 0) (safety : 0 ≤ s.surplus) :
    s.pools + s.claims ≤ s.vault := by
  simp_all [MathCan.FlashAccounting.Backed, MathCan.FlashAccounting.FlashBook.claims, MathCan.FlashAccounting.FlashBook.deltas, MathCan.FlashAccounting.FlashBook.pools, MathCan.FlashAccounting.FlashBook.surplus, MathCan.FlashAccounting.FlashBook.vault]
  omega
#eval IO.println "RAM03_AXIOMS_BEGIN"
#print axioms terminal_vault_solvency
#eval IO.println "RAM03_AXIOMS_END"
end MathCan.FlashAccounting
