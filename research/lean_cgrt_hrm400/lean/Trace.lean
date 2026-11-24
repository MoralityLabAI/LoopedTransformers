/-
Tactic tracing for real Lean source (e.g. Mathlib files).

`traceFile` elaborates a whole file with Lean's own frontend (imports from its header), walks
the info trees, and writes one JSON line per source tactic: the enclosing declaration, the
source text and position, the goals before and after as Lean pretty-prints them, and,
for a sample of steps, the outcome of candidate tactics run on the same goal in the
environment immediately before the enclosing source command. Nonportable goals
are explicitly unscored. Source trace rows retain their original pretty-printed goals. The reply summarises the run: steps, declarations with their axioms, and
error messages (for the checker status of each declaration).
-/
import Lean
open Lean Elab Meta

namespace LGymTrace

/-- Structural nodes whose info duplicates their children: sequences, the `by` keyword
(an atom, kind `by`) and the bare `·` token; the `· tac` block itself is kept. -/
def skipKinds : List SyntaxNodeKind :=
  [``Lean.Parser.Tactic.tacticSeq, ``Lean.Parser.Tactic.tacticSeq1Indented,
   ``Lean.Parser.Tactic.tacticSeqBracketed, ``Lean.Parser.Term.byTactic, nullKind,
   Name.mkSimple "by", `Lean.cdotTk]

/-- Select mode: run these tactics on the main goal of the source tactic at this position. -/
structure Sel where
  line : Nat
  col : Nat
  tactics : Array String
  deriving FromJson

structure Opts where
  path : String
  out : String
  candidates : Array String := #[]
  candEvery : Nat := 0            -- run candidates on every k-th step (0: never)
  candHeartbeats : Nat := 20000   -- per candidate, in thousands (Lean's unit)
  maxPP : Nat := 4000
  select : Array Sel := #[]       -- non-empty: write only the selected positions, with certificates
  selMap : Std.HashMap (Nat × Nat) (Array String) := {}

def truncate (s : String) (n : Nat) : String := if s.length > n then s.take n ++ "…" else s

unsafe def enableInitsUnsafe : IO Unit := enableInitializersExecution

/-- `importModules` (via `withImporting`) switches initializer execution off again when any
import finishes, so it must be re-enabled before every header import, or `[init]`
declarations of imported modules (Aesop options, Mathlib tactics) never run. -/
@[implemented_by enableInitsUnsafe] opaque enableInits : IO Unit

/-- Heartbeats are counted per process; give each candidate its own budget from now. -/
def withBudget (limit : Nat) (x : MetaM α) : MetaM α := do
  let hb ← IO.getNumHeartbeats
  withTheReader Core.Context (fun c => { c with initHeartbeats := hb, maxHeartbeats := limit * 1000 }) x

def ppGoals (ctx : ContextInfo) (mctx : MetavarContext) (gs : List MVarId) (maxPP : Nat) : IO (Array Json) :=
  ({ ctx with mctx }).runMetaM {} <| withBudget 200000 do
    gs.toArray.mapM fun g => do return Json.str (truncate (toString (← ppGoal g)) maxPP)

/-- Universe metavariables left by the enclosing elaboration (a proof inside a definition whose
universes are fixed only after its body is elaborated) become universe parameters of the
certified theorem. The kernel then checks the statement at every universe, so this can reject a
proof that needs a particular universe, never accept a wrong one. -/
def levelMVarsToParams (e : Expr) : Expr :=
  e.replaceLevel fun l =>
    if l.hasMVar then
      some (l.replace fun l' => match l' with
        | .mvar id => some (.param (Name.mkSimple s!"u_lgym_{id.name.hash}"))
        | _ => none)
    else none

/-- Kernel certificate for a goal a tactic closed: abstract the goal's local context over the
proof term, add it as a theorem (the kernel typechecks it) and collect its axioms. The
environment change is discarded with the candidate's MetaM state. -/
def certifyClosed (g : MVarId) : MetaM Json := do
  let pf ← instantiateMVars (mkMVar g)
  if pf.hasSorry then return Json.mkObj [("kernel", false), ("why", "sorry")]
  if pf.hasExprMVar then return Json.mkObj [("kernel", false), ("why", "unassigned metavariables")]
  g.withContext do
    let ty ← instantiateMVars (← g.getType)
    if ty.hasExprMVar then return Json.mkObj [("kernel", false), ("why", "goal has metavariables")]
    -- Abstract only the hypotheses the statement and proof depend on (transitively): unused
    -- hypotheses can still mention metavariables of the enclosing elaboration.
    let ids := (← getLCtx).getFVarIds
    let mut need := (collectFVars (collectFVars {} ty) pf).fvarSet
    for id in ids.reverse do
      if need.contains id then
        let d ← id.getDecl
        -- Recursive theorem elaboration can carry a local auxiliary declaration
        -- standing for the theorem itself. Abstracting it as a hypothesis would
        -- certify a weaker, circular statement. Check the transitive dependency
        -- closure of BOTH the target and the proof, including local let values.
        if d.isAuxDecl then
          return Json.mkObj [("kernel", false), ("why", "depends on auxiliary recursion declaration")]
        let mut s := collectFVars {} (← instantiateMVars d.type)
        if let some v := d.value? then s := collectFVars s (← instantiateMVars v)
        need := s.fvarIds.foldl (·.insert ·) need
    let xs := (ids.filter need.contains).map mkFVar
    let ty0 ← instantiateMVars (← mkForallFVars xs ty)
    let pf0 ← instantiateMVars (← mkLambdaFVars xs pf)
    let gen := ty0.hasLevelMVar || pf0.hasLevelMVar
    let ty' := levelMVarsToParams ty0
    let pf' := levelMVarsToParams pf0
    if ty'.hasMVar || pf'.hasMVar then
      return Json.mkObj [("kernel", false), ("why", "context has metavariables")]
    let lps := (collectLevelParams (collectLevelParams {} ty') pf').params
    let n := Name.mkSimple "_lgym_cert"
    try
      addDecl (.thmDecl { name := n, levelParams := lps.toList, type := ty', value := pf' })
    catch e =>
      return Json.mkObj [("kernel", false), ("why", Json.str (truncate (← e.toMessageData.toString) 200))]
    let ax ← collectAxioms n
    let base := [("kernel", Json.bool true), ("axioms", toJson (ax.map toString))]
    return Json.mkObj (if gen then base ++ [("levels_generalized", Json.bool true)] else base)

/-- Universe names a tactic may mention (`foo.{u, v}`): the universe parameters occurring in the
goal and its local context. The info tree keeps no `universe` declarations, so without these
an explicit universe in a source tactic is an unknown level. -/
def goalLevelNames (g : MVarId) : MetaM (List Name) := g.withContext do
  let mut s : CollectLevelParams.State := {}
  s := collectLevelParams s (← instantiateMVars (← g.getType))
  for d in (← getLCtx) do
    unless d.isImplementationDetail do
      s := collectLevelParams s (← instantiateMVars d.type)
  return s.params.toList

/-- Whether any tactic could complete the goal under `certifyClosed`: it must be a proposition
(a data goal, such as an instance or a definition's body, cannot be a theorem), and its
statement, with the hypotheses that statement depends on, must have no expression
metavariables (a goal still tied to an unfinished elaboration has no standalone statement). -/
def certifiable (g : MVarId) : MetaM (Bool × Bool) := g.withContext do
  let ty ← instantiateMVars (← g.getType)
  let prop ← try isProp ty catch _ => pure false
  let mut closed := !ty.hasExprMVar
  let mut need := (collectFVars {} ty).fvarSet
  for id in (← getLCtx).getFVarIds.reverse do
    if need.contains id then
      let d ← id.getDecl
      if d.isAuxDecl then closed := false
      let t ← instantiateMVars d.type
      if t.hasExprMVar then closed := false
      let mut s := collectFVars {} t
      if let some v := d.value? then
        let v ← instantiateMVars v
        if v.hasExprMVar then closed := false
        s := collectFVars s v
      need := s.fvarIds.foldl (·.insert ·) need
  return (prop, closed)

/-- Conservative portability check. The tactic info tree has the environment
AFTER the command; checking kernel axioms there cannot detect self-reference.
Re-use only a goal whose instantiated target and local context are expressible
in the exact preceding environment. Never fall back to the later environment
for candidate execution. -/
def stateConsts (g : MVarId) : MetaM (Array Name) := g.withContext do
  let mut cs := (← instantiateMVars (← g.getType)).getUsedConstants
  for d in (← getLCtx) do
    cs := cs ++ (← instantiateMVars d.type).getUsedConstants
    if let some v := d.value? then
      cs := cs ++ (← instantiateMVars v).getUsedConstants
  return cs

def beforeContext (envBefore : Environment) (ctx : ContextInfo)
    (ti : TacticInfo) (g : MVarId) : IO (Option ContextInfo) := do
  let some decl := ctx.parentDecl? | return none
  if decl.isAnonymous || envBefore.contains decl then return none
  let portable ← try
    ({ ctx with mctx := ti.mctxBefore }).runMetaM {} (withBudget 400000 do
      let cs ← stateConsts g
      return cs.all envBefore.contains)
    catch _ => pure false
  if portable then return some { ctx with env := envBefore }
  return none

def runCandidate (ctx : ContextInfo) (ti : TacticInfo) (g : MVarId) (tac : String) (hbLimit : Nat)
    (certify : Bool := false) : IO Json :=
  ({ ctx with mctx := ti.mctxBefore }).runMetaM {} do
    let env ← getEnv
    let some decl := ctx.parentDecl? |
      return Json.mkObj [("tactic", Json.str tac), ("ok", false), ("error", "scope: missing declaration")]
    if decl.isAnonymous || env.contains decl then
      return Json.mkObj [("tactic", Json.str tac), ("ok", false), ("error", "scope: declaration visible")]
    match Parser.runParserCategory env `tactic tac with
    | .error e => return Json.mkObj [("tactic", Json.str tac), ("ok", false), ("error", Json.str s!"parse: {e}")]
    | .ok stx =>
      modifyThe Core.State fun st => { st with messages := {} }
      let lvls ← goalLevelNames g
      -- tryCatchRuntimeEx: a heartbeat or recursion-depth timeout is a runtime exception,
      -- which plain try/catch lets through; here it is the candidate's failure.
      tryCatchRuntimeEx
        (do
          let gs ← g.withContext <| withBudget hbLimit <|
            (Term.withLevelNames lvls <| Term.withoutErrToSorry (Tactic.run g (Tactic.evalTactic stx))).run'
          let msgs : MessageLog := (← getThe Core.State).messages
          if msgs.hasErrors then
            let t ← msgs.toList.filterMapM fun (m : Message) => do
              if m.severity == MessageSeverity.error then return some (← m.data.toString) else return none
            return Json.mkObj [("tactic", Json.str tac), ("ok", false),
                               ("error", Json.str (truncate (String.intercalate "; " t) 240))]
          if certify && gs.isEmpty then
            return Json.mkObj [("tactic", Json.str tac), ("ok", true), ("goals_after", toJson gs.length),
                               ("cert", ← certifyClosed g)]
          return Json.mkObj [("tactic", Json.str tac), ("ok", true), ("goals_after", toJson gs.length)])
        (fun e => do
          let txt ← (show Exception from e).toMessageData.toString
          return Json.mkObj [("tactic", Json.str tac), ("ok", false), ("error", Json.str (truncate txt 240))])

structure St where
  steps : Nat := 0
  decls : Std.HashMap Name Nat := {}
  selDone : Std.HashSet (Nat × Nat) := {}

/-- Select mode: the outermost source tactic at a selected position gets the requested
tactics run on its main goal, each closing result with a kernel certificate. -/
def visitSelected (envBefore : Environment) (o : Opts) (input : String) (h : IO.FS.Handle) (st : IO.Ref St) (ctx : ContextInfo)
    (ti : TacticInfo) (p e : String.Pos) : IO Unit := do
  let pos := ctx.fileMap.toPosition p
  let key := (pos.line, pos.column)
  let some tacs := o.selMap[key]? | return
  if ti.stx.isAtom || (← st.get).selDone.contains key then return
  st.modify fun s => { s with selDone := s.selDone.insert key }
  let before ← ppGoals ctx ti.mctxBefore ti.goalsBefore o.maxPP
  let mut res := #[]
  let mut cert := (false, false)
  let mut scoped := false
  let decl := ctx.parentDecl?.getD .anonymous
  if let some g := ti.goalsBefore.head? then
    if let some safeCtx ← beforeContext envBefore ctx ti g then
      scoped := true
      cert ← try ({ safeCtx with mctx := ti.mctxBefore }).runMetaM {} (certifiable g) catch _ => pure (false, false)
      for c in tacs do
        res := res.push (← runCandidate safeCtx ti g c o.candHeartbeats (certify := true))
  let row := Json.mkObj [
    ("scope_version", "pre_command_v1"),
    ("env_scope", Json.str (if scoped then "before" else "unavailable")),
    ("decl_in_env_before", toJson (envBefore.contains decl)),
    ("decl", Json.str (ctx.parentDecl?.getD .anonymous).toString), ("line", toJson pos.line),
    ("col", toJson pos.column), ("tactic", Json.str (truncate (input.extract p e) 2000)),
    ("goals_before", Json.arr before), ("goals", toJson ti.goalsBefore.length),
    ("main_prop", toJson cert.1), ("main_closed", toJson cert.2), ("results", Json.arr res)]
  h.putStrLn row.compress

partial def visit (envBefore : Environment) (o : Opts) (input : String) (h : IO.FS.Handle) (st : IO.Ref St)
    (ctx? : Option ContextInfo) : InfoTree → IO Unit
  | .context pci t => visit envBefore o input h st (pci.mergeIntoOuter? ctx?) t
  | .hole _ => pure ()
  | .node i cs => do
    if let (some ctx, .ofTacticInfo ti) := (ctx?, i) then
      let stx := ti.stx
      if !skipKinds.contains stx.getKind && !o.selMap.isEmpty then
        if let (some p, some e) := (stx.getPos? (canonicalOnly := true), stx.getTailPos? (canonicalOnly := true)) then
          visitSelected envBefore o input h st ctx ti p e
      else if !skipKinds.contains stx.getKind then
        if let (some p, some e) := (stx.getPos? (canonicalOnly := true), stx.getTailPos? (canonicalOnly := true)) then
          let n := (← st.get).steps
          let decl := ctx.parentDecl?.getD .anonymous
          st.modify fun s => { s with steps := n + 1, decls := s.decls.insert decl (s.decls.getD decl 0 + 1) }
          let pos := ctx.fileMap.toPosition p
          let before ← ppGoals ctx ti.mctxBefore ti.goalsBefore o.maxPP
          let after ← ppGoals ctx ti.mctxAfter ti.goalsAfter o.maxPP
          let mut cands := #[]
          let mut scoped := false
          if o.candEvery > 0 && n % o.candEvery == 0 then
            if let some g := ti.goalsBefore.head? then
              if let some safeCtx ← beforeContext envBefore ctx ti g then
                scoped := true
                for c in o.candidates do
                  cands := cands.push (← runCandidate safeCtx ti g c o.candHeartbeats)
          let row := Json.mkObj [
            ("scope_version", "pre_command_v1"),
            ("env_scope", Json.str (if scoped then "before" else "unavailable_or_not_queried")),
            ("decl_in_env_before", toJson (envBefore.contains decl)),
            ("decl", Json.str decl.toString), ("line", toJson pos.line), ("col", toJson pos.column),
            ("tactic", Json.str (truncate (input.extract p e) 2000)),
            ("kind", Json.str stx.getKind.toString), ("elaborator", Json.str ti.elaborator.toString),
            ("goals_before", Json.arr before), ("goals_after", Json.arr after),
            ("candidates", Json.arr cands)]
          h.putStrLn row.compress
    for c in cs do visit envBefore o input h st ctx? c

def traceFile (o : Opts) : IO Json := do
  let o := { o with selMap := o.select.foldl (fun m s => m.insert (s.line, s.col) s.tactics) {} }
  let t0 ← IO.monoMsNow
  let input ← IO.FS.readFile o.path
  let inputCtx := Parser.mkInputContext input o.path
  let (header, parserState, messages) ← Parser.parseHeader inputCtx
  enableInits
  let (env, messages) ← processHeader header {} messages inputCtx (trustLevel := 0)
  let cmdState := { Command.mkState env messages {} with infoState := { enabled := true } }
  let h ← IO.FS.Handle.mk o.out .write
  let st ← IO.mkRef ({} : St)
  -- Inspect new info trees immediately after each command, but run candidates
  -- in the saved environment BEFORE that command (Fable's command-wise replay).
  let mut s : Frontend.State := { commandState := cmdState, parserState, cmdPos := parserState.pos }
  let mut done := false
  let mut elabMs : Nat := 0
  while !done do
    let envBefore := s.commandState.env
    let n0 := s.commandState.infoState.trees.size
    let c0 ← IO.monoMsNow
    let (d, s') ← (Frontend.processCommand.run { inputCtx }).run s
    let c1 ← IO.monoMsNow
    elabMs := elabMs + (c1 - c0)
    s := s'
    done := d
    let trees := s.commandState.infoState.trees
    for i in [n0:trees.size] do
      visit envBefore o inputCtx.input h st none trees[i]!
  h.flush
  let finalEnv := s.commandState.env
  let fileMap := inputCtx.fileMap
  let mut declsJ := #[]
  for (d, k) in (← st.get).decls.toList do
    let axioms ← if d.isAnonymous || !finalEnv.contains d then pure #[] else
      (do let (r, _) ← (collectAxioms d : CoreM (Array Name)).toIO { fileName := o.path, fileMap } { env := finalEnv }
          pure (r.map toString)) <|> pure #["<axiom collection failed>"]
    declsJ := declsJ.push (Json.mkObj [("decl", Json.str d.toString), ("steps", toJson k),
                                       ("in_env", toJson (finalEnv.contains d)), ("axioms", toJson axioms)])
  let mut errs := #[]
  for m in s.commandState.messages.toList do
    if m.severity == .error then
      errs := errs.push (Json.mkObj [("line", toJson m.pos.line), ("col", toJson m.pos.column),
                                     ("text", Json.str (truncate (← m.data.toString) 300))])
  let t2 ← IO.monoMsNow
  return Json.mkObj [("ok", true), ("scope_version", "pre_command_v1"),
                     ("steps", toJson (← st.get).steps), ("decls", Json.arr declsJ),
                     ("errors", Json.arr errs), ("elab_ms", toJson elabMs),
                     ("trace_ms", toJson (t2 - t0 - elabMs))]

end LGymTrace
