"""Only model-visible public fields; pinned native Nanbeige chat rendering."""
from __future__ import annotations
import json
from pathlib import Path

from .cohort import PUBLIC_KEYS
from .source import format_series

DEFINITIONS=(
    "A formal power series lists coefficients from x^0 up to the specified degree.\n"
    "Every integer uses a fixed signed digit representation: P means plus, M means minus,\n"
    "followed by three separated decimal digits, e.g. P 0 0 3 = 3 and M 0 1 2 = -12.\n"
    "I(f)=f; N(f)=f(-x); S(f)=f(x^2); R(f)=1/f as a formal reciprocal series.\n"
)


def body(row:dict)->str:
    if set(row)!=PUBLIC_KEYS:
        raise ValueError("only exact public source fields are permitted")
    m=row["m"];degree=row["degree"];kind=row["kind"]
    if m not in (3,5) or degree not in (2*m-1,2*m,2*m+2):
        raise ValueError("unregistered exponent or evidence degree")
    if kind=="classification":
        if len(row["demos"])!=2 or row["query"]:
            raise ValueError("two demonstrations only")
        blocks=[]
        for idx,example in enumerate(row["demos"],1):
            if len(example["input"])!=degree+1:
                raise ValueError("input degree mismatch")
            output=example["output"]
            if output is None:
                rendered="[redacted]"
            else:
                if len(output)!=degree+1:
                    raise ValueError("output degree mismatch")
                rendered=format_series(output)
            blocks.append(f"Example {idx} input: {format_series(example['input'])}\n"
                          f"Example {idx} output: {rendered}\n")
        return (DEFINITIONS+
            f"Two examples use the same one operator and show coefficients through degree {degree}.\n"
            +"".join(blocks)+
            "Identify the operator from I,N,S,R. Reply with one uppercase letter, without explanation.")
    if kind=="coefficient":
        if row["demos"] or len(row["query"])!=degree+1:
            raise ValueError("numeric query must be the only model-visible source")
        return (DEFINITIONS+
            f"Input f has coefficients through degree {degree}:\n"
            f"{format_series(row['query'])}\n"
            f"What is the coefficient of x^{degree} in the formal series 1/f?\n"
            "Return one signed decimal integer inside square brackets, e.g. [12] or [-12].")
    raise ValueError("unregistered task kind")


def encode(tokenizer,snapshot,row:dict)->list[int]:
    t=tokenizer.tok
    cfg=json.loads((Path(snapshot)/"tokenizer_config.json").read_text(encoding="utf-8"))
    t.chat_template=cfg["chat_template"]
    t.bos_token=cfg["bos_token"]
    t.eos_token=cfg["eos_token"]
    text=t.apply_chat_template([{"role":"user","content":body(row)}],
                               tokenize=False,add_generation_prompt=True,
                               enable_thinking=False)
    tail="\nOperator:" if row["kind"]=="classification" else "\nCoefficient:"
    return t.encode(text+tail,add_special_tokens=False)
