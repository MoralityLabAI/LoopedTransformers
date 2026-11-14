import Lean

open Lean Elab Tactic Meta

namespace RAM03

structure Request where
  id : Nat := 0
  actions : Array String := #[]
  stop : Bool := false
  deriving FromJson

private def directory : TacticM System.FilePath := do
  let some path ← IO.getEnv "RAM03_MAILBOX" | throwError "RAM03_MAILBOX missing"
  return path

private def emit (id : Nat) (j : Json) : TacticM Unit := do
  let dir ← directory
  let path := dir / s!"response-{id}.json"
  let temporary := dir / s!"response-{id}.tmp"
  IO.FS.writeFile temporary j.compress
  IO.FS.rename temporary path

private def observe : TacticM Json := do
  pruneSolvedGoals
  let goals ← getGoals
  let mut text : Array Json := #[]
  let mut constants : Array Name := #[]
  for goal in goals do
    text := text.push (toJson (← ppGoal goal).pretty)
    let decl ← goal.getDecl
    constants := constants ++ decl.type.getUsedConstants
    for localDecl in decl.lctx do
      constants := constants ++ localDecl.type.getUsedConstants
  let env ← getEnv
  let mut definitions : Array String := #[]
  for name in constants do
    if name.toString.startsWith "MathCan." then
      if let some (.defnInfo _) := env.find? name then
        if !definitions.contains name.toString then
          definitions := definitions.push name.toString
  return Json.mkObj [("accepted", toJson true), ("goals", toJson text),
    ("definitions", toJson definitions)]

elab "ram03_probe_server" : tactic => do
  let initial ← Tactic.saveState
  let dir ← directory
  emit 0 (Json.mkObj [("ready", toJson true), ("state", ← observe)])
  let mut id := 1
  repeat
    let path := dir / s!"request-{id}.json"
    while !(← path.pathExists) do IO.sleep 5
    let line ← IO.FS.readFile path
    let request := do
      let json ← Json.parse line
      fromJson? json (α := Request)
    match request with
    | .error error => emit id (Json.mkObj [("accepted", toJson false), ("error", toJson error)])
    | .ok request =>
      if request.stop then break
      initial.restore
      try
        Core.withCurrHeartbeats do
          for action in request.actions do
            let parsed := Parser.runParserCategory (← getEnv) `tactic action
            match parsed with
            | .error error => throwError error
            | .ok stx => evalTactic stx
          if (← Core.getMessageLog).hasErrors then throwError "tactic logged an error"
          let result ← observe
          emit id (Json.mkObj [("id", toJson request.id), ("result", result)])
      catch error =>
        emit id (Json.mkObj [("id", toJson request.id), ("result", Json.mkObj
          [("accepted", toJson false), ("error", toJson (← error.toMessageData.toString))])])
    id := id + 1
  initial.restore
  evalTactic (← `(tactic| all_goals sorry))

end RAM03
