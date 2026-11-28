"""Build and score prompt-fixed InkyNano matrix cases."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def jsonl(path):
    with path.open(encoding="utf8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def render(case):
    if "rendered_prompt" in case:
        return case["rendered_prompt"]
    if case["mode"] == "outfill":
        return case["prefix"]
    if case["mode"] == "infill":
        return ("Fill the single missing section. Return only its replacement.\n"
                f"<before>\n{case['prefix']}\n</before>\n"
                f"<after>\n{case['suffix']}\n</after>\n<replacement>\n")
    raise ValueError("unknown generation mode")


def cases_and_labels(path):
    records = list(jsonl(path))
    ids, families = set(), {}
    for case in records:
        if case["id"] in ids:
            raise ValueError(f"duplicate matrix case {case['id']}")
        ids.add(case["id"])
        split = case["split"]
        if split not in ("train", "dev") or families.setdefault(case["root"], split) != split:
            raise ValueError(f"invalid or overlapping root split {case['root']}")
        if (case["mode"] not in ("infill", "outfill") or not case["prefix"]
                or (case["mode"] == "infill") != bool(case["suffix"])
                or "rendered_prompt" in case and not (
                    isinstance(case["rendered_prompt"], str) and case["rendered_prompt"].strip()
                )):
            raise ValueError(f"invalid prompt parts {case['id']}")
    return records


def requests(root, task, prefix, suffix, label, split, max_tokens):
    for mode in ("outfill", "infill"):
        yield {
            "id": f"{root}/{mode}", "root": root, "task": task, "mode": mode, "split": split,
            "prefix": prefix, "suffix": suffix if mode == "infill" else "",
            "max_tokens": max_tokens, "label": label,
        }


def sudoku_solutions(grid, limit, randomizer=None):
    """Choose the most constrained blank and stop after limit completions."""
    empty = [(r, c) for r in range(9) for c in range(9) if not grid[r][c]]
    groups = ([set(row) - {0} for row in grid],
              [{grid[r][c] for r in range(9)} - {0} for c in range(9)],
              [{grid[r][c] for r in range(i // 3 * 3, i // 3 * 3 + 3)
                for c in range(i % 3 * 3, i % 3 * 3 + 3)} - {0} for i in range(9)])
    digits, answers = set(range(1, 10)), []

    def search():
        if not empty:
            answers.append([row.copy() for row in grid])
            return
        options = []
        for i, (r, c) in enumerate(empty):
            used = groups[0][r], groups[1][c], groups[2][r // 3 * 3 + c // 3]
            available = digits.difference(*used)
            if not available:
                return
            options.append((len(available), i, available, used))
            if len(available) == 1:
                break
        _, index, available, used = min(options)
        r, c = empty.pop(index)
        moves = sorted(available)
        if randomizer:
            randomizer.shuffle(moves)
        for value in moves:
            grid[r][c] = value
            for group in used:
                group.add(value)
            search()
            for group in used:
                group.remove(value)
            if len(answers) >= limit:
                break
        grid[r][c] = 0
        empty.insert(index, (r, c))

    search()
    return answers


def grid_text(grid):
    """A row of single-digit cells with no separator between cells."""
    return "\n".join("".join(map(str, row)) for row in grid)


def sudoku_cases(train, dev, seed, compact=False):
    for i in range(train + dev):
        root_seed = seed * 100_000 + i
        rng = random.Random(root_seed)
        solution, = sudoku_solutions([[0] * 9 for _ in range(9)], 1, rng)
        givens = [row.copy() for row in solution]
        cells = rng.sample(range(81), 81)
        removed = 0
        for index in cells:
            r, c = divmod(index, 9)
            value, givens[r][c] = givens[r][c], 0
            if len(sudoku_solutions(givens, 2)) != 1:
                givens[r][c] = value
            else:
                removed += 1
                if removed == 36:
                    break
        givens_text = grid_text(givens) if compact else canonical(givens)
        prefix = ("Solve this 9x9 Sudoku. 0 is a blank. Respect the givens and make every row, "
                  'column, and 3x3 box contain digits 1 through 9. Return JSON {"grid":[...]} only.\n'
                  + givens_text + "\nAnswer:\n")
        label = {"grid": solution, "givens": givens}
        yield from requests(f"sudoku/{root_seed}", "sudoku", prefix, "End of answer.", label,
                            "train" if i < train else "dev", 256)


def arc_cases(training, train, dev, seed, compact=False):
    paths = sorted(training.glob("*.json"), key=lambda p: digest([seed, p.stem]))
    if len(paths) < train + dev:
        raise ValueError("insufficient ARC public training roots")
    for position, path in enumerate(paths[:train + dev]):
        task = json.loads(path.read_text(encoding="utf8"))
        tests = [pair["input"] for pair in task["test"]]
        instruction = ("Infer the grid transformation from the examples. Predict each test grid in order. "
                       'Return JSON {"attempt_1":[grid,...],"attempt_2":[grid,...]} only.\n')
        if compact:
            examples = [f"Example {i} input:\n{grid_text(pair['input'])}\nOutput:\n{grid_text(pair['output'])}"
                        for i, pair in enumerate(task["train"], 1)]
            test_inputs = [f"Test {i} input:\n{grid_text(grid)}" for i, grid in enumerate(tests, 1)]
            prefix = (instruction + "Each line is a row. Each digit is one cell.\n" +
                      "\n".join(examples + test_inputs) + "\nAnswer:\n")
        else:
            prefix = instruction + f"Examples: {canonical(task['train'])}\nTest inputs: {canonical(tests)}\nAnswer:\n"
        answer = [pair["output"] for pair in task["test"]]
        yield from requests(f"arc-agi-2-training/{path.stem}", "arc-agi-2", prefix,
                            "End of answer.", answer, "train" if position < train else "dev", 384)


def hilbert_storyworld_cases(world_path, train, dev, seed):
    world = json.loads(world_path.read_text(encoding="utf8"))
    plan = world["plan"]
    scenes = sorted(enumerate(world["scenes"]), key=lambda pair: digest([seed, pair[1]["id"]]))
    if len(scenes) < train + dev:
        raise ValueError("insufficient Hilbert Storyworld encounters")
    encounters_per_act, remainder = divmod(len(scenes), len(plan["acts"]))
    if remainder or not encounters_per_act:
        raise ValueError("Hilbert Storyworld does not align encounters with acts")
    operators = sorted(world["math"]["operators"])
    clues = sorted(world["math"]["clues"])
    cast = [[entry[1], entry[3]] for entry in plan["cast"]]
    for position, (index, scene) in enumerate(scenes[:train + dev]):
        advances = [1, 2] if int(scene["id"].removeprefix("enc_")) % 6 == 3 else [1]
        if sorted({option[4] for option in scene["options"]}) != advances:
            raise ValueError(f"published encounter {scene['id']} violates its advance rule")
        contract = {"id": scene["id"], "title": scene["title"], "allowed_operators": operators,
                    "allowed_clues": clues, "required_advances": advances}
        context = {"world": plan["title"], "premise": plan["premise"], "cast": cast,
                   "act": plan["acts"][index // encounters_per_act], **contract}
        prefix = ("Author one encounter as JSON with id, title, text, and options. Copy id and title exactly. "
                  "Write two to four short paragraphs of original prose. Include three distinct choice labels. "
                  "Each option is [label, voiced_response, operator, clue_or_null, advance]. Use only the "
                  "allowed operators and clues and cover exactly the required advances.\n"
                  f"{canonical(context)}\nEncounter:\n")
        label = {"required_keys": ["id", "title", "text", "options"], "hilbert_encounter": contract}
        yield from requests(f"storyworld-hilbert/{scene['id']}", "storyworld", prefix, "End of encounter.",
                            label, "train" if position < train else "dev", 512)


def response_body(text):
    text = text.rsplit("</think>", 1)[-1].strip()
    native = text.removeprefix("<|message_model|>")
    if native.startswith("<|content_thinking|>"):
        _, _, native = native.partition("<|end_message|>")
        native = native.strip().removeprefix("<|message_model|>")
    if native.startswith("<|content_text|>"):
        text = native.removeprefix("<|content_text|>").strip()
        text = text.removesuffix("<|content_model_end_sampling|>").rstrip()
        text = text.removesuffix("<|end_message|>").rstrip()
    if text.startswith("```") and text.endswith("```"):
        text = text.partition("\n")[2][:-3].strip()
    return text


def result_json(text):
    try:
        value = json.loads(response_body(text))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def integer_grid(value):
    return (isinstance(value, list) and bool(value) and isinstance(value[0], list) and bool(value[0])
            and all(isinstance(row, list) and len(row) == len(value[0])
                    and all(type(cell) is int for cell in row) for row in value))


def compact_response_grid(text):
    lines = response_body(text).splitlines()
    if (not lines or not all(len(line) == len(lines[0]) and line.isascii() and line.isdecimal()
                             for line in lines)):
        return None
    return [[int(cell) for cell in line] for line in lines]


def hilbert_encounter_valid(result, contract):
    prose = result.get("text")
    options = result.get("options")
    if (any(result.get(key) != contract[key] for key in ("id", "title"))
            or not isinstance(prose, str) or not prose.strip()
            or not isinstance(options, list) or len(options) != 3):
        return False
    for row in options:
        if (not isinstance(row, list) or len(row) != 5
                or not all(isinstance(value, str) and value.strip() for value in row[:2])
                or row[2] not in contract["allowed_operators"]
                or (row[3] is not None and row[3] not in contract["allowed_clues"])
                or type(row[4]) is not int):
            return False
    return (len({row[0].strip().casefold() for row in options}) == 3
            and sorted({row[4] for row in options}) == contract["required_advances"])


def score(case, text):
    result = result_json(text)
    label = case["label"]
    if case["task"] == "sudoku":
        grid = None if result is None else result.get("grid")
        parsed = integer_grid(grid)
        correct = parsed and grid == label["grid"]
        return {"parse_ok": parsed, "correct": correct,
                "exact_grid_match": correct or compact_response_grid(text) == label["grid"]}
    if case["task"] == "arc-agi-2":
        guesses = [result.get(key) for key in ("attempt_1", "attempt_2")] if result else []
        valid = [attempt for attempt in guesses if isinstance(attempt, list)
                 and len(attempt) == len(label) and all(integer_grid(g) for g in attempt)]
        tests = [any(attempt[i] == expected for attempt in valid) for i, expected in enumerate(label)]
        correct = all(tests)
        return {"parse_ok": bool(valid), "correct": correct, "test_pass_at_2": tests,
                "exact_grid_match": correct or (len(label) == 1 and compact_response_grid(text) == label[0])}
    if case["task"] == "storyworld" and isinstance(label, dict) and "required_keys" in label:
        keys_present = result is not None and all(key in result for key in label["required_keys"])
        scores = {"parse_ok": result is not None, "schema_keys_present": keys_present}
        if "hilbert_encounter" in label:
            scores["storyworld_structure_ok"] = keys_present and hilbert_encounter_valid(
                result, label["hilbert_encounter"])
            prose = result.get("text") if result else None
            paragraphs = [part for part in prose.split("\n\n") if part.strip()] if isinstance(prose, str) else []
            scores["storyworld_paragraph_target_ok"] = 2 <= len(paragraphs) <= 4
        return scores
    return {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arc-public-training", type=Path, required=True)
    storyworld = p.add_mutually_exclusive_group()
    storyworld.add_argument("--storyworld", type=Path, help="Goose cases already in matrix JSONL format")
    storyworld.add_argument("--hilbert-storyworld", type=Path, help="derive encounter-authoring cases from a published Hilbert world")
    p.add_argument("--train-per-task", type=int, default=12)
    p.add_argument("--dev-per-task", type=int, default=4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--compact-grids", action="store_true", help="render rows as consecutive single-digit cells")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if min(a.train_per_task, a.dev_per_task) <= 0:
        p.error("positive train/dev counts are required")
    records = list(sudoku_cases(a.train_per_task, a.dev_per_task, a.seed, a.compact_grids))
    records.extend(arc_cases(a.arc_public_training, a.train_per_task, a.dev_per_task, a.seed, a.compact_grids))
    if a.storyworld:
        story = cases_and_labels(a.storyworld)
        if any(case["task"] != "storyworld" for case in story):
            p.error("storyworld input must contain only storyworld cases")
        records.extend(story)
    if a.hilbert_storyworld:
        records.extend(hilbert_storyworld_cases(a.hilbert_storyworld, a.train_per_task, a.dev_per_task, a.seed))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x", encoding="utf8", newline="\n") as output:
        for case in records:
            output.write(canonical(case) + "\n")
    cases_and_labels(a.out)
    print(f"{len(records)} cases from {len({c['root'] for c in records})} roots")


if __name__ == "__main__":
    main()
