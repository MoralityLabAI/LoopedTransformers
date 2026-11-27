/-
lgym: a JSON-lines proof-state server over Lean 4 core.

Python never decides whether a tactic applies. It sends a theorem statement,
Lean elaborates it and reports goals; it sends tactics, Lean runs them and
reports the resulting goals or the error message; at the end Lean's kernel
checks the assembled proof term (`kernel`). A separate file compile
(`lcgrt.certify`) is the certificate of record.

Requests (one JSON object per line on stdin; one reply per line on stdout):
  {"op":"init","stmt":"∀ (P0 P1 : Prop) (h0 : P0) (r0 : P0 → P1), P1"}
  {"op":"expand","sid":3}           -- try `apply h` for every hypothesis h and `constructor`
  {"op":"run","sid":3,"tactic":"apply r0"}
  {"op":"kernel","sid":7}           -- kernel-check the finished proof of a closed state
  {"op":"drop","sids":[1,2]}        -- forget states
  {"op":"stats"}
  {"op":"trace","path":"Mathlib/Logic/Basic.lean","out":"steps.jsonl","candidates":["simp"],"cand_every":5}
                                    -- elaborate a source file, write its tactic states (Trace.lean)
-/
import Lean
import Trace
import Session
open Lean Meta Elab

structure GState where
  saved : Meta.SavedState
  goals : List MVarId
  root : Expr
  rootType : Expr

structure Server where
  env : Environment
  states : IO.Ref (Std.HashMap Nat GState)
  next : IO.Ref Nat
  seenCtx : IO.Ref (Std.HashMap UInt64 Unit)
  pp : IO.Ref Bool
  requests : IO.Ref Nat

partial def propJson (e : Expr) : MetaM Json := do
  let e ← instantiateMVars e
  match e with
  | .fvar id => return Json.mkObj [("a", Json.str (← id.getDecl).userName.toString)]
  | .sort .zero => return Json.mkObj [("sort", "Prop")]
  | .forallE _ d b _ =>
    if b.hasLooseBVars then
      return Json.mkObj [("raw", Json.str (toString (← ppExpr e)))]
    else
      return Json.mkObj [("imp", Json.arr #[← propJson d, ← propJson b])]
  | .const ``True _ => return Json.mkObj [("true", Json.null)]
  | .const ``False _ => return Json.mkObj [("false", Json.null)]
  | _ =>
    if e.isAppOfArity ``And 2 then
      return Json.mkObj [("and", Json.arr #[← propJson (e.getArg! 0), ← propJson (e.getArg! 1)])]
    else if e.isAppOfArity ``Or 2 then
      return Json.mkObj [("or", Json.arr #[← propJson (e.getArg! 0), ← propJson (e.getArg! 1)])]
    else if e.isAppOfArity ``Not 1 then
      return Json.mkObj [("not", ← propJson (e.getArg! 0))]
    else
      return Json.mkObj [("raw", Json.str (toString (← ppExpr e)))]

def hypsJson (g : MVarId) : MetaM (Array Json × UInt64) := g.withContext do
  let mut out := #[]
  let mut h : UInt64 := 7
  for d in (← getLCtx) do
    if d.isImplementationDetail then continue
    let ty ← propJson d.type
    let row := Json.mkObj [("name", Json.str d.userName.toString), ("type", ty)]
    h := mixHash h (hash row.compress)
    out := out.push row
  return (out, h)

def goalJson (srv : Server) (g : MVarId) : MetaM Json := g.withContext do
  let (hyps, ctxId) ← hypsJson g
  let target ← propJson (← g.getType)
  let seen ← srv.seenCtx.get
  let mut fields := [("target", target), ("ctx", Json.str (toString ctxId))]
  if ← srv.pp.get then
    fields := fields ++ [("pp", Json.str (toString (← ppGoal g)))]
  unless seen.contains ctxId do
    srv.seenCtx.modify (·.insert ctxId ())
    fields := fields ++ [("hyps", Json.arr hyps)]
  return Json.mkObj fields

def runMeta (srv : Server) (x : MetaM α) : IO α := do
  -- The heartbeat counter is process-global; the limit is per request.
  let ctx : Core.Context := { fileName := "<lgym>", fileMap := default,
                              initHeartbeats := ← IO.getNumHeartbeats,
                              maxHeartbeats := 20000 * 1000, options := {} }
  -- SavedState.restore does not restore the name generator, so every request
  -- needs its own prefix; otherwise new goal mvars reuse ids already assigned
  -- in the restored context and `apply` reports them as solved.
  let req ← srv.requests.modifyGet fun n => (n, n + 1)
  let st : Core.State := { env := srv.env, ngen := { namePrefix := .num `_lgym req } }
  let (a, _) ← (x.run' {} {}).toIO ctx st
  return a

def storeState (srv : Server) (s : GState) : IO Nat := do
  let id ← srv.next.get
  srv.next.set (id + 1)
  srv.states.modify (·.insert id s)
  return id

def stateReply (srv : Server) (sid : Nat) (goals : List MVarId) : MetaM Json := do
  let gs ← goals.mapM (goalJson srv)
  return Json.mkObj [("sid", toJson sid), ("goals", Json.arr gs.toArray), ("closed", toJson goals.isEmpty)]

def elabStmt (stmt : String) : TermElabM Expr := do
  let env ← getEnv
  match Parser.runParserCategory env `term stmt with
  | .error e => throwError "parse error: {e}"
  | .ok stx =>
    let ty ← Term.elabType stx
    Term.synthesizeSyntheticMVarsNoPostponing
    instantiateMVars ty

def opInit (srv : Server) (stmt : String) : IO Json := do
  runMeta srv do
    let ty ← (elabStmt stmt).run'
    let root ← mkFreshExprMVar ty (kind := .syntheticOpaque)
    let n := getIntrosSize ty
    let (_, g) ← root.mvarId!.introNP n
    let saved ← saveState
    let sid ← storeState srv { saved, goals := [g], root, rootType := ty }
    stateReply srv sid [g]

def getState (srv : Server) (sid : Nat) : IO GState := do
  match (← srv.states.get)[sid]? with
  | some s => return s
  | none => throw <| IO.userError s!"unknown sid {sid}"

/-- First line of Lean's message plus the unification pair, without the goal dump. -/
def errText (e : Exception) : MetaM String := do
  let s ← e.toMessageData.toString
  let lines := (s.splitOn "\n").take 4
  let t := String.intercalate " " (lines.map String.trim)
  return if t.length > 240 then t.take 240 else t

/-- Run one candidate on the main goal; Lean's own result goals come first. -/
def applyOnMain (s : GState) (tac : String) (f : MVarId → MetaM (List MVarId))
    (render : Bool := true) : MetaM (Except String (List MVarId)) := do
  s.saved.restore
  match s.goals with
  | [] => return .error "no goals"
  | g :: rest =>
    try
      let new ← f g
      return .ok (new ++ rest)
    catch e =>
      -- Rendering pretty-prints the whole goal; skip it for unsampled failures.
      if render then return .error s!"{tac}: {← errText e}" else return .error ""

/-- Move goal `i` to the front (the `pick_goal` permutation); goals are untouched. -/
def opFocus (srv : Server) (sid i : Nat) : IO Json := do
  let s ← getState srv sid
  if h : i < s.goals.length then
    let g := s.goals[i]
    let goals := g :: s.goals.eraseIdx i
    let child ← storeState srv { s with goals }
    runMeta srv do
      s.saved.restore
      return Json.mkObj [("ok", true), ("state", ← stateReply srv child goals)]
  else
    return Json.mkObj [("ok", false), ("error", Json.str "goal index out of range")]

/-- `errors` bounds how many failure messages are returned (all failures are listed). -/
def opExpand (srv : Server) (sid : Nat) (maxErrors : Nat) : IO Json := do
  let s ← getState srv sid
  runMeta srv do
    s.saved.restore
    let some g := s.goals.head? | return Json.mkObj [("sid", toJson sid), ("candidates", Json.arr #[])]
    let hyps ← g.withContext do
      let mut acc := #[]
      for d in (← getLCtx) do
        if d.isImplementationDetail then continue
        if ← isProp d.type then acc := acc.push (d.userName, d.fvarId)
      return acc
    let mut out := #[]
    let mut errs := 0
    let mut cands : Array (String × (MVarId → MetaM (List MVarId))) := #[]
    for (n, fv) in hyps do
      cands := cands.push (s!"apply {n}", fun g => g.apply (mkFVar fv))
    cands := cands.push ("constructor", fun g => g.constructor)
    for (tac, f) in cands do
      match ← applyOnMain s tac f (render := errs < maxErrors) with
      | .error msg =>
        let fields := [("tactic", Json.str tac), ("ok", (false : Json))]
        let fields := if errs < maxErrors then fields ++ [("error", Json.str msg)] else fields
        errs := errs + 1
        out := out.push (Json.mkObj fields)
      | .ok goals =>
        let saved ← saveState
        let child ← storeState srv { s with saved, goals }
        let rep ← stateReply srv child goals
        out := out.push (Json.mkObj [("tactic", Json.str tac), ("ok", true), ("state", rep)])
    return Json.mkObj [("sid", toJson sid), ("candidates", Json.arr out)]

def opRun (srv : Server) (sid : Nat) (tac : String) : IO Json := do
  let s ← getState srv sid
  runMeta srv do
    s.saved.restore
    let env ← getEnv
    match Parser.runParserCategory env `tactic tac with
    | .error e => return Json.mkObj [("ok", false), ("error", Json.str s!"parse error: {e}")]
    | .ok stx =>
      -- Elaboration errors inside a tactic are normally logged and replaced by
      -- `sorry`; a logged error must count as failure, never as progress.
      modifyThe Core.State fun st => { st with messages := {} }
      let r ← applyOnMain s tac fun g => do
        let gs ← (Term.withoutErrToSorry (Tactic.run g (Tactic.evalTactic stx))).run'
        let msgs := (← getThe Core.State).messages
        if msgs.hasErrors then
          let texts ← msgs.toList.filterMapM fun m =>
            if m.severity == .error then return some (← m.data.toString) else return none
          throwError "{String.intercalate "; " texts}"
        return gs
      match r with
      | .error msg => return Json.mkObj [("ok", false), ("error", Json.str msg)]
      | .ok goals =>
        let saved ← saveState
        let child ← storeState srv { s with saved, goals }
        return Json.mkObj [("ok", true), ("state", ← stateReply srv child goals)]

def opKernel (srv : Server) (sid : Nat) : IO Json := do
  let s ← getState srv sid
  runMeta srv do
    s.saved.restore
    unless s.goals.isEmpty do
      return Json.mkObj [("ok", false), ("error", Json.str "open goals remain")]
    let pf ← instantiateMVars s.root
    if pf.hasMVar then
      return Json.mkObj [("ok", false), ("error", Json.str "proof term has metavariables")]
    let name := `lgym_kernel_check
    let decl := Declaration.thmDecl { name, levelParams := [], type := s.rootType, value := pf }
    try
      addDecl decl
      let axioms ← collectAxioms name
      return Json.mkObj [("ok", true), ("axioms", toJson (axioms.map toString)),
                         ("term", Json.str (toString (← ppExpr pf)))]
    catch e =>
      return Json.mkObj [("ok", false), ("error", Json.str (← errText e))]

def handle (srv : Server) (j : Json) : IO Json := do
  let op ← IO.ofExcept (j.getObjValAs? String "op")
  match op with
  | "init" =>
    srv.pp.set ((j.getObjValAs? Bool "pp").toOption.getD false)
    opInit srv (← IO.ofExcept (j.getObjValAs? String "stmt"))
  | "expand" => opExpand srv (← IO.ofExcept (j.getObjValAs? Nat "sid"))
                  ((j.getObjValAs? Nat "errors").toOption.getD 1000000)
  | "focus" => opFocus srv (← IO.ofExcept (j.getObjValAs? Nat "sid")) (← IO.ofExcept (j.getObjValAs? Nat "goal"))
  | "run" => opRun srv (← IO.ofExcept (j.getObjValAs? Nat "sid")) (← IO.ofExcept (j.getObjValAs? String "tactic"))
  | "kernel" => opKernel srv (← IO.ofExcept (j.getObjValAs? Nat "sid"))
  | "drop" =>
    let ids ← IO.ofExcept (j.getObjValAs? (Array Nat) "sids")
    for i in ids do srv.states.modify (·.erase i)
    return Json.mkObj [("ok", true)]
  | "reset" =>
    srv.states.set {}
    srv.seenCtx.set {}
    return Json.mkObj [("ok", true)]
  | "trace" =>
    -- This standalone server exposes the legacy tracer solely for source sampling.
    -- All speculative actions MUST use the pre-command session interface.
    if !((j.getObjValAs? (Array LGymTrace.Sel) "select").toOption.getD #[]).isEmpty ||
        (j.getObjValAs? Nat "cand_every").toOption.getD 0 != 0 then
      throw <| IO.userError "legacy candidate tracing disabled; use open/step/cert"
    LGymTrace.traceFile {
      path := ← IO.ofExcept (j.getObjValAs? String "path"), out := ← IO.ofExcept (j.getObjValAs? String "out"),
      candidates := (j.getObjValAs? (Array String) "candidates").toOption.getD #[],
      candEvery := (j.getObjValAs? Nat "cand_every").toOption.getD 0,
      candHeartbeats := (j.getObjValAs? Nat "cand_heartbeats").toOption.getD 20000,
      select := (j.getObjValAs? (Array LGymTrace.Sel) "select").toOption.getD #[] }
  | "open" =>
    let pos ← IO.ofExcept (j.getObjValAs? (Array (Array Nat)) "positions")
    LGymSession.openFile (← IO.ofExcept (j.getObjValAs? String "path")) (pos.map fun a => (a[0]!, a[1]!))
  | "step" =>
    LGymSession.stepState (← IO.ofExcept (j.getObjValAs? Nat "sid")) (← IO.ofExcept (j.getObjValAs? String "tactic"))
      ((j.getObjValAs? Nat "hb").toOption.getD 200)
  | "premises" =>
    LGymSession.premises (← IO.ofExcept (j.getObjValAs? Nat "sid")) ((j.getObjValAs? Nat "max_rw").toOption.getD 40)
      ((j.getObjValAs? Nat "max_apply").toOption.getD 20)
  | "graph" => LGymSession.graph (← IO.ofExcept (j.getObjValAs? Nat "sid"))
  | "lemma_graph" =>
    LGymSession.lemmaGraphs (← IO.ofExcept (j.getObjValAs? Nat "sid")) (← IO.ofExcept (j.getObjValAs? (Array String) "names"))
  | "cert" => LGymSession.certState (← IO.ofExcept (j.getObjValAs? Nat "sid"))
  | "keep" => LGymSession.dropStates (← IO.ofExcept (j.getObjValAs? (Array Nat) "sids"))
  | "stats" =>
    return Json.mkObj [("states", toJson (← srv.states.get).size), ("next", toJson (← srv.next.get))]
  | other => throw <| IO.userError s!"unknown op {other}"

unsafe def main (args : List String) : IO UInt32 := do
  -- Imported modules with `initialize` declarations (Aesop, Mathlib tactics) must have their
  -- initializers run in this process, as Lean's own frontend does (the call is `unsafe`).
  enableInitializersExecution
  initSearchPath (← findSysroot)
  let imports := (args.map fun m => ({ module := m.toName } : Import)).toArray
  let imports := if imports.isEmpty then #[{ module := `Init }] else imports
  -- loadExts: the parser/notation extensions (`∧`, `→`) must be live to elaborate text.
  let env ← importModules imports {} (trustLevel := 0) (loadExts := true)
  let srv : Server := { env, states := ← IO.mkRef {}, next := ← IO.mkRef 0, seenCtx := ← IO.mkRef {},
                        pp := ← IO.mkRef false, requests := ← IO.mkRef 0 }
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  stdout.putStrLn (Json.mkObj [("ready", true), ("scope_version", "pre_command_session_v1"), ("lean", Json.str Lean.versionString)]).compress
  stdout.flush
  repeat
    let line ← stdin.getLine
    if line.isEmpty then break
    let line := line.trimRight
    if line.isEmpty then continue
    let h0 ← IO.getNumHeartbeats
    let t0 ← IO.monoMsNow
    let reply ← try
        match Json.parse line with
        | .error e => pure (Json.mkObj [("error", Json.str s!"bad json: {e}")])
        | .ok j => handle srv j
      catch e => pure (Json.mkObj [("error", Json.str (toString e))])
    let h1 ← IO.getNumHeartbeats
    let t1 ← IO.monoMsNow
    let reply := (reply.setObjVal! "request_hb_raw" (toJson (h1 - h0))).setObjVal! "request_ms" (toJson (t1 - t0))
    stdout.putStrLn reply.compress
    stdout.flush
  return 0
