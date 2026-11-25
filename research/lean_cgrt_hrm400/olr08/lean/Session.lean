/-
Stepwise proof sessions on real source states (OLR-08 gym server).

`open`  elaborates a source file once and stores, for each selected source position, the goal as
        it stood there together with the environment of that point of the file.
`step`  runs one tactic on the main goal of a stored state and stores the resulting state.
`cert`  kernel-checks the finished proof of the original goal and reports its axioms.
Python never decides whether a tactic applies or whether a proof is complete.
-/
import Lean
import Trace
open Lean Elab Meta

namespace LGymSession
open LGymTrace

structure TState where
  ctx : ContextInfo
  mctx : MetavarContext
  goals : List MVarId
  root : MVarId
  depth : Nat := 0

initialize sessions : IO.Ref (Std.HashMap Nat TState) ← IO.mkRef {}
initialize nextSid : IO.Ref Nat ← IO.mkRef 0
initialize reqNo : IO.Ref Nat ← IO.mkRef 0

/-- Every request gets its own name-generator prefix, so metavariables made in one step can never
reuse an id made in another step or by the file's own elaboration. -/
def freshCtx (s : TState) : IO ContextInfo := do
  let req ← reqNo.modifyGet fun n => (n, n + 1)
  return { s.ctx with mctx := s.mctx, ngen := { namePrefix := .num `_lgym8 req, idx := 1 } }

def runAt (ctx : ContextInfo) (mctx : MetavarContext) (x : MetaM α) : IO α :=
  ({ ctx with mctx }).runMetaM {} (withBudget 400000 x)

def goalsJson (goals : List MVarId) (maxPP : Nat) : MetaM (Array Json) :=
  goals.toArray.mapM fun m => do return Json.str (truncate (toString (← ppGoal m)) maxPP)

/-- Every constant the goal and its local context mention. -/
def stateConsts (g : MVarId) : MetaM (Array Name) := g.withContext do
  let mut cs := (← instantiateMVars (← g.getType)).getUsedConstants
  for d in (← getLCtx) do
    cs := cs ++ (← instantiateMVars d.type).getUsedConstants
    if let some v := d.value? then cs := cs ++ (← instantiateMVars v).getUsedConstants
  return cs

/-- The info tree of a declaration carries the environment as it is AFTER the declaration's
command, so the theorem being proved (and its `simp` attribute) is visible to tactics run there.
`envBefore` is the environment that preceded the command. A state is opened in `envBefore`
whenever everything it mentions already exists there; otherwise it is reported as not portable
and keeps the later environment (`"env":"after"`), which a benchmark must exclude. -/
partial def visitOpen (envBefore : Environment) (sel : Std.HashSet (Nat × Nat)) (done : IO.Ref (Std.HashSet (Nat × Nat)))
    (acc : IO.Ref (Array Json)) (ctx? : Option ContextInfo) : InfoTree → IO Unit
  | .context pci t => visitOpen envBefore sel done acc (pci.mergeIntoOuter? ctx?) t
  | .hole _ => pure ()
  | .node i cs => do
    if let (some ctx, .ofTacticInfo ti) := (ctx?, i) then
      let stx := ti.stx
      if !skipKinds.contains stx.getKind && !stx.isAtom then
        if let some p := stx.getPos? (canonicalOnly := true) then
          let pos := ctx.fileMap.toPosition p
          let key := (pos.line, pos.column)
          if sel.contains key && !(← done.get).contains key then
            done.modify (·.insert key)
            if let some g := ti.goalsBefore.head? then
              -- Names introduced by `induction`, `cases` and `intro` stay addressable.
              let ctxA := { ctx with options := ctx.options.setBool `tactic.hygienic false }
              let decl := ctx.parentDecl?.getD .anonymous
              let cs ← try runAt ctxA ti.mctxBefore (stateConsts g) catch _ => pure #[decl]
              let portable := !decl.isAnonymous && !envBefore.contains decl && cs.all envBefore.contains
              if portable then
                let ctx' := { ctxA with env := envBefore }
                let cert ← try runAt ctx' ti.mctxBefore (certifiable g) catch _ => pure (false, false)
                let info ← try runAt ctx' ti.mctxBefore (stateInfo g 300) catch _ => pure Json.null
                let pp ← try runAt ctx' ti.mctxBefore (goalsJson [g] 4000) catch _ => pure #[]
                if cert.1 && cert.2 && ti.goalsBefore.length == 1 then
                  let sid ← nextSid.modifyGet fun n => (n, n + 1)
                  sessions.modify (·.insert sid { ctx := ctx', mctx := ti.mctxBefore, goals := [g], root := g })
                  acc.modify (·.push (Json.mkObj [
                    ("line", toJson pos.line), ("col", toJson pos.column),
                    ("decl", Json.str decl.toString), ("sid", toJson sid),
                    ("env", "before"), ("scope_version", "pre_command_session_v1"),
                    ("decl_in_env", toJson (ctx'.env.contains decl)),
                    ("source_goals", toJson ti.goalsBefore.length), ("goals", Json.arr pp),
                    ("main_prop", toJson cert.1), ("main_closed", toJson cert.2), ("info", info)]))
                else
                  acc.modify (·.push (Json.mkObj [("line", toJson pos.line), ("col", toJson pos.column),
                    ("decl", Json.str decl.toString), ("env", "unavailable"), ("reason", "uncertifiable")]))
              else
                acc.modify (·.push (Json.mkObj [("line", toJson pos.line), ("col", toJson pos.column),
                  ("decl", Json.str decl.toString), ("env", "unavailable"), ("reason", "nonportable_before_command")]))
    for c in cs do visitOpen envBefore sel done acc ctx? c

def openFile (path : String) (positions : Array (Nat × Nat)) : IO Json := do
  sessions.set {}
  let input ← IO.FS.readFile path
  let inputCtx := Parser.mkInputContext input path
  let (header, parserState, messages) ← Parser.parseHeader inputCtx
  enableInits
  let (env, messages) ← processHeader header {} messages inputCtx (trustLevel := 0)
  let cmdState := { Command.mkState env messages {} with infoState := { enabled := true } }
  let sel : Std.HashSet (Nat × Nat) := positions.foldl (·.insert ·) {}
  let done ← IO.mkRef ({} : Std.HashSet (Nat × Nat))
  let acc ← IO.mkRef (#[] : Array Json)
  -- One command at a time, so the environment before each command is known.
  let mut st : Frontend.State := { commandState := cmdState, parserState, cmdPos := parserState.pos }
  let mut fin := false
  let mut ncmd := 0
  while !fin do
    let envBefore := st.commandState.env
    let n0 := st.commandState.infoState.trees.size
    let (d, st') ← (Frontend.processCommand.run { inputCtx }).run st
    st := st'
    fin := d
    ncmd := ncmd + 1
    let trees := st.commandState.infoState.trees
    for i in [n0:trees.size] do
      visitOpen envBefore sel done acc none trees[i]!
  let errs := st.commandState.messages.toList.filter (·.severity == .error)
  return Json.mkObj [("ok", true), ("states", Json.arr (← acc.get)), ("errors", toJson errs.length),
                     ("commands", toJson ncmd)]

def stepState (sid : Nat) (tac : String) (hb : Nat) : IO Json := do
  let some s := (← sessions.get)[sid]? | return Json.mkObj [("ok", false), ("error", "unknown sid")]
  let ctx ← freshCtx s
  let t0 ← IO.monoMsNow
  let h0 ← IO.getNumHeartbeats
  let r : Json × Option (MetavarContext × List MVarId × Environment) ← ctx.runMetaM {} <| withBudget (hb + 400000) do
    match s.goals with
    | [] => return (Json.mkObj [("ok", false), ("error", "no goals")], none)
    | g :: rest =>
      match Parser.runParserCategory (← getEnv) `tactic tac with
      | .error e => return (Json.mkObj [("ok", false), ("error", Json.str s!"parse: {e}")], none)
      | .ok stx =>
        modifyThe Core.State fun st => { st with messages := {} }
        let lvls ← goalLevelNames g
        tryCatchRuntimeEx
          (do
            let gs ← g.withContext <| withBudget hb <|
              (Term.withLevelNames lvls <| Term.withoutErrToSorry (Tactic.run g (Tactic.evalTactic stx))).run'
            let msgs : MessageLog := (← getThe Core.State).messages
            if msgs.hasErrors then
              let t ← msgs.toList.filterMapM fun (m : Message) => do
                if m.severity == MessageSeverity.error then return some (← m.data.toString) else return none
              return (Json.mkObj [("ok", false), ("error", Json.str (truncate (String.intercalate "; " t) 240))], none)
            -- A goal assigned by unification while a sibling was solved is no longer an obligation.
            let goals ← (gs ++ rest).filterM fun m => do return !(← m.isAssigned)
            let pp ← goalsJson goals 2000
            let info ← match goals with
              | m :: _ => stateInfo m 300
              | [] => pure Json.null
            return (Json.mkObj [("ok", true), ("ngoals", toJson goals.length), ("goals", Json.arr pp),
                                ("info", info)], some (← getMCtx, goals, ← getEnv)))
          (fun e => do
            let txt ← (show Exception from e).toMessageData.toString
            return (Json.mkObj [("ok", false), ("error", Json.str (truncate txt 240))], none))
  let t1 ← IO.monoMsNow
  let h1 ← IO.getNumHeartbeats
  let j := (r.1.setObjVal! "ms" (toJson (t1 - t0))).setObjVal! "hb" (toJson ((h1 - h0) / 1000))
  match r.2 with
  | some (mctx, goals, env) =>
    -- Tactics add auxiliary declarations (`simp` proofs, `split` splitters); later steps and the
    -- final certificate must see them.
    let sid' ← nextSid.modifyGet fun n => (n, n + 1)
    sessions.modify (·.insert sid' { s with ctx := { s.ctx with env }, mctx, goals, depth := s.depth + 1 })
    return j.setObjVal! "sid" (toJson sid')
  | none => return j

/-- Exact premise proposals for the main goal, from Lean's own indices: lemmas whose left or right
side matches a subterm (the `rw?` discrimination tree) and lemmas whose conclusion matches the
goal (the `exact?`/`apply?` tree). Only declarations visible at this point of the file. -/
def premiseCands (g : MVarId) (maxRw maxApply : Nat) : MetaM Json := g.withContext do
  let target ← instantiateMVars (← g.getType)
  let moduleRef ← Lean.Meta.Rewrites.createModuleTreeRef
  let rw ← Lean.Meta.Rewrites.rewriteCandidates #[] moduleRef target {}
  let mut rws : Array Json := #[]
  for (l, symm, w) in rw do
    if rws.size ≥ maxRw then break
    if let .inr n := l then
      unless n.isInternal do
        rws := rws.push (Json.mkObj [("name", Json.str n.toString), ("symm", toJson symm), ("w", toJson w)])
  let ap ← try Lean.Meta.LibrarySearch.libSearchFindDecls target catch _ => pure #[]
  let mut aps : Array Json := #[]
  for (n, _) in ap do
    if aps.size ≥ maxApply then break
    unless n.isInternal do
      aps := aps.push (Json.mkObj [("name", Json.str n.toString)])
  return Json.mkObj [("rw", Json.arr rws), ("apply", Json.arr aps),
                     ("rw_total", toJson rw.size), ("apply_total", toJson ap.size)]

def premises (sid maxRw maxApply : Nat) : IO Json := do
  let some s := (← sessions.get)[sid]? | return Json.mkObj [("ok", false), ("error", "unknown sid")]
  let some g := s.goals.head? | return Json.mkObj [("ok", false), ("error", "no goals")]
  let ctx ← freshCtx s
  let t0 ← IO.monoMsNow
  let r ← try ctx.runMetaM {} (withBudget 5000 (premiseCands g maxRw maxApply))
    catch e => return Json.mkObj [("ok", false), ("error", Json.str (toString e))]
  let t1 ← IO.monoMsNow
  return (r.setObjVal! "ok" (Json.bool true)).setObjVal! "ms" (toJson (t1 - t0))

/-! ### Term graph export

The reader's input: the hash-consed graph of every open obligation. One node per distinct subterm,
so a local variable is a single node shared by all of its occurrences and by sibling obligations.
Edges run from a subterm to the term that contains it, labelled with the role it plays there. -/

structure GB where
  ids : Std.HashMap Expr Nat := {}
  nodes : Array Json := #[]
  edges : Array Json := #[]

abbrev GM := StateT GB MetaM

def mkNode (e : Expr) (fields : List (String × Json)) : GM Nat := do
  let i := (← get).nodes.size
  modify fun s => { s with ids := s.ids.insert e i, nodes := s.nodes.push (Json.mkObj fields) }
  return i

def addEdge (child parent : Nat) (role : String) : GM Unit :=
  modify fun s => { s with edges := s.edges.push (Json.arr #[toJson child, toJson parent, Json.str role]) }

partial def addExpr (e : Expr) (fuel : Nat) : GM Nat := do
  if let some i := (← get).ids[e]? then return i
  if fuel == 0 then return (← mkNode e [("k", "cut")])
  match e with
  | .const n _ =>
    let c := match (← getEnv).find? n with
      | some (.defnInfo _) => "def"
      | some (.thmInfo _) => "thm"
      | some (.ctorInfo _) => "ctor"
      | some (.inductInfo _) => "ind"
      | some (.recInfo _) => "rec"
      | _ => "other"
    mkNode e [("k", "const"), ("n", Json.str n.toString), ("c", c)]
  | .fvar id => mkNode e [("k", "fvar"), ("n", Json.str (← id.getDecl).userName.toString)]
  | .mvar id => mkNode e [("k", "mvar"), ("n", Json.str id.name.toString)]
  | .bvar i => mkNode e [("k", "bvar"), ("i", toJson i)]
  | .sort _ => mkNode e [("k", "sort")]
  | .lit (.natVal v) => mkNode e [("k", "nat"), ("v", toJson (min v 1000))]
  | .lit (.strVal _) => mkNode e [("k", "str")]
  | .mdata _ b => addExpr b fuel
  | .proj s idx b =>
    let i ← mkNode e [("k", "proj"), ("n", Json.str s.toString), ("i", toJson idx)]
    addEdge (← addExpr b (fuel - 1)) i "arg0"
    return i
  | .app .. =>
    let f := e.getAppFn
    let args := e.getAppArgs
    let i ← mkNode e [("k", "app")]
    addEdge (← addExpr f (fuel - 1)) i "fn"
    for j in [0:args.size] do
      addEdge (← addExpr args[j]! (fuel - 1)) i s!"arg{min j 8}"
    return i
  | .lam _ t b _ =>
    let i ← mkNode e [("k", "lam")]
    addEdge (← addExpr t (fuel - 1)) i "btype"
    addEdge (← addExpr b (fuel - 1)) i "body"
    return i
  | .forallE _ t b _ =>
    let i ← mkNode e [("k", if b.hasLooseBVars then "pi" else "arrow")]
    addEdge (← addExpr t (fuel - 1)) i "btype"
    addEdge (← addExpr b (fuel - 1)) i "body"
    return i
  | .letE _ t v b _ =>
    let i ← mkNode e [("k", "let")]
    addEdge (← addExpr t (fuel - 1)) i "btype"
    addEdge (← addExpr v (fuel - 1)) i "arg0"
    addEdge (← addExpr b (fuel - 1)) i "body"
    return i

def obligationGraph (g : MVarId) : GM Json := do
  let lctx := (← g.getDecl).lctx
  let ty ← instantiateMVars (← g.getType)
  let t ← withLCtx lctx #[] (addExpr ty 64)
  let mut hyps : Array Json := #[]
  for d in lctx do
    if d.isImplementationDetail || d.isAuxDecl then continue
    let dt ← instantiateMVars d.type
    let v ← withLCtx lctx #[] (addExpr (mkFVar d.fvarId) 64)
    let tn ← withLCtx lctx #[] (addExpr dt 64)
    addEdge tn v "typeof"
    if let some value := d.value? then
      let vn ← withLCtx lctx #[] (addExpr (← instantiateMVars value) 64)
      addEdge vn v "value"
    let p ← try withLCtx lctx #[] (isProp dt) catch _ => pure false
    hyps := hyps.push (Json.mkObj [("node", toJson v), ("type", toJson tn),
      ("name", Json.str d.userName.toString), ("acc", toJson (!d.userName.hasMacroScopes)), ("prop", toJson p)])
  return Json.mkObj [("target", toJson t), ("hyps", Json.arr hyps)]

def stateGraph (goals : List MVarId) : MetaM Json := do
  let (obs, gb) ← (goals.toArray.mapM obligationGraph).run {}
  return Json.mkObj [("nodes", Json.arr gb.nodes), ("edges", Json.arr gb.edges), ("obligations", Json.arr obs)]

def graph (sid : Nat) : IO Json := do
  let some s := (← sessions.get)[sid]? | return Json.mkObj [("ok", false), ("error", "unknown sid")]
  let ctx ← freshCtx s
  let r ← try ctx.runMetaM {} (withBudget 5000 (stateGraph s.goals))
    catch e => return Json.mkObj [("ok", false), ("error", Json.str (toString e))]
  return r.setObjVal! "ok" (Json.bool true)

/-- Term graphs of library statements, for the lemma cache. Each statement is a closed term, so
its bound variables appear as `bvar` nodes under `pi` and `arrow` nodes. Names that do not resolve
in this state's environment are skipped. -/
def lemmaGraphs (sid : Nat) (names : Array String) : IO Json := do
  let some s := (← sessions.get)[sid]? | return Json.mkObj [("ok", false), ("error", "unknown sid")]
  let ctx ← freshCtx s
  let r ← try ctx.runMetaM {} (withBudget 5000 do
        let env ← getEnv
        let mut out : Array Json := #[]
        for n in names do
          if let some ci := env.find? n.toName then
            let (t, gb) ← (addExpr ci.type 64).run {}
            out := out.push (Json.mkObj [("name", Json.str n), ("root", toJson t),
              ("nodes", Json.arr gb.nodes), ("edges", Json.arr gb.edges)])
        return Json.arr out)
    catch e => return Json.mkObj [("ok", false), ("error", Json.str (toString e))]
  return Json.mkObj [("ok", true), ("lemmas", r)]

/-- A proof that mentions the recursion variable of the enclosing declaration assumes the theorem
it is proving. `certifyClosed` would abstract that variable as a hypothesis and the kernel would
accept the weakened statement, so it is refused here. -/
def usesAuxDecl (g : MVarId) : MetaM Bool := g.withContext do
  let pf ← instantiateMVars (mkMVar g)
  let fv := (collectFVars {} pf).fvarSet
  for d in (← getLCtx) do
    if d.isAuxDecl && fv.contains d.fvarId then return true
  return false

/-- Kernel certificate of the original goal of a finished state. -/
def certState (sid : Nat) : IO Json := do
  let some s := (← sessions.get)[sid]? | return Json.mkObj [("ok", false), ("error", "unknown sid")]
  unless s.goals.isEmpty do return Json.mkObj [("ok", false), ("error", "open goals remain")]
  let ctx ← freshCtx s
  let c ← try ctx.runMetaM {} (withBudget 20000 do
        if ← usesAuxDecl s.root then
          return Json.mkObj [("kernel", false), ("why", "self-reference through the recursion variable")]
        certifyClosed s.root)
    catch e => pure (Json.mkObj [("kernel", false), ("why", Json.str (toString e))])
  return Json.mkObj [("ok", true), ("cert", c), ("depth", toJson s.depth)]

def dropStates (keep : Array Nat) : IO Json := do
  let k : Std.HashSet Nat := keep.foldl (·.insert ·) {}
  sessions.modify fun m => m.fold (fun acc i s => if k.contains i then acc.insert i s else acc) {}
  return Json.mkObj [("ok", true), ("states", toJson (← sessions.get).size)]

end LGymSession
