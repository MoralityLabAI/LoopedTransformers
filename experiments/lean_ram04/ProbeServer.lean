import Lean.Elab.Tactic.Basic
import Lean.Elab.Tactic.BuiltinTactic
import Lean.Elab.Tactic.RCases
import Lean.Elab.Tactic.Induction
import Lean.Elab.Tactic.Simp
import Lean.Elab.Tactic.Omega
import Lean.Elab.Deriving.FromToJson

open Lean Elab Tactic Meta

namespace RAM04

structure Request where
  id : Nat := 0
  actions : Array String := #[]
  stop : Bool := false
  deriving FromJson

private def directory : TacticM System.FilePath := do
  let some path ← IO.getEnv "RAM04_MAILBOX" | throwError "RAM04_MAILBOX missing"
  return path

private def retryIO {α : Type} (action : IO α) : Nat → IO α
  | 0 => action
  | n+1 => do
    try action catch _ =>
      IO.sleep 5
      retryIO action n

private def emit (id : Nat) (j : Json) : TacticM Unit := do
  let dir ← directory
  let path := dir / s!"response-{id}.json"
  let temporary := dir / s!"response-{id}.tmp"
  retryIO (do IO.FS.writeFile temporary j.compress; IO.FS.rename temporary path) 400

private def observe : TacticM Json := do
  pruneSolvedGoals
  let goals ← getGoals
  let mut text : Array Json := #[]
  let mut constants : Array Name := #[]
  let mut locals : Array Json := #[]
  let mut first := true
  for goal in goals do
    text := text.push (toJson (← ppGoal goal).pretty)
    let decl ← goal.getDecl
    constants := constants ++ decl.type.getUsedConstants
    for localDecl in decl.lctx do
      constants := constants ++ localDecl.type.getUsedConstants
    if first then
      locals ← goal.withContext do
        let mut rows : Array Json := #[]
        for localDecl in decl.lctx do
          let name := localDecl.userName
          if !localDecl.isImplementationDetail && !name.isInaccessibleUserName && !name.hasMacroScopes then
            let typ ← instantiateMVars localDecl.type
            let reduced ← whnf typ
            let domain ← match reduced with
              | .forallE _ dom _ _ => pure (← ppExpr dom).pretty
              | _ => pure ""
            let head := reduced.getAppFn.constName?.getD .anonymous
            let bodyHead := reduced.getForallBody.getAppFn.constName?.getD .anonymous
            rows := rows.push (Json.mkObj [("name", toJson name.toString),
              ("type", toJson (← ppExpr typ).pretty), ("prop", toJson (← isProp typ)),
              ("head", toJson head.toString), ("function", toJson reduced.isForall),
              ("domain", toJson domain), ("rewritable", toJson (bodyHead == ``Eq || bodyHead == ``Iff))])
        return rows
      first := false
  let env ← getEnv
  let mut definitions : Array String := #[]
  for name in constants do
    if name.toString.startsWith "MathCan." then
      if let some (.defnInfo _) := env.find? name then
        if !definitions.contains name.toString then
          definitions := definitions.push name.toString
  return Json.mkObj [("accepted", toJson true), ("goals", toJson text),
    ("definitions", toJson definitions), ("locals", toJson locals)]

private def boundedRequest (action : TacticM Json) : TacticM Json := do
  let heartbeats ← IO.getNumHeartbeats
  controlAt CoreM fun runInBase =>
    withReader (fun (ctx : Core.Context) =>
      { ctx with initHeartbeats := heartbeats, maxHeartbeats := 100000000 })
      (runInBase action)

elab "ram04_probe_server" : tactic => do
  let initial ← Tactic.saveState
  let dir ← directory
  emit 0 (Json.mkObj [("ready", toJson true), ("state", ← observe)])
  let mut id := 1
  repeat
    let path := dir / s!"request-{id}.json"
    while !(← path.pathExists) do IO.sleep 5
    let line ← retryIO (IO.FS.readFile path) 400
    let request := do
      let json ← Json.parse line
      fromJson? json (α := Request)
    match request with
    | .error error => emit id (Json.mkObj [("accepted", toJson false), ("error", toJson error)])
    | .ok request =>
      if request.stop then break
      initial.restore
      Core.resetMessageLog
      let result ←
        try
          boundedRequest do
            for action in request.actions do
              let parsed := Parser.runParserCategory (← getEnv) `tactic action
              match parsed with
              | .error error => throwError error
              | .ok stx => evalTactic stx
            if (← Core.getMessageLog).hasErrors then throwError "tactic logged an error"
            observe
        catch error =>
          pure (Json.mkObj [("accepted", toJson false), ("error", toJson (← error.toMessageData.toString))])
      emit id (Json.mkObj [("id", toJson request.id), ("result", result)])
    id := id + 1
  initial.restore
  Core.resetMessageLog
  evalTactic (← `(tactic| all_goals sorry))

end RAM04
