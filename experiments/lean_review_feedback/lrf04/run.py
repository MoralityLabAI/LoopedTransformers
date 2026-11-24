"""Native LRF-04 integration run; all real proof calls are archived."""
import argparse
import json
import subprocess
from pathlib import Path
import controller as C
A=C.A
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mathcan",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();mathcan=a.mathcan.resolve();out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    version=subprocess.check_output(["lake","env","lean","--version"],cwd=mathcan,text=True).strip()
    assert "4.35.0" in version
    A.checkout(mathcan)
    native=A.native(mathcan,(HERE/"Conservative.lean").read_text(),out/"conservative.lean",60.)
    assert native["status"]=="compiled",native
    required={"LRF04."+n for n in ("conservative_sound","states_complete","compatible_matches_demo","collision_denied","unknown_denied")}
    assert set(native["reports"])==required and all(set(x)<=A.ALLOW for x in native["reports"].values()),native
    lines=(out/"conservative.log").read_text().splitlines()
    marker=[x.removeprefix("LRF04_NATIVE ") for x in lines if x.startswith("LRF04_NATIVE ")]
    assert len(marker)==1,lines
    table=json.loads(marker[0])
    expected=[C.policy(m,o,x)[0] for m in C.MODES for o in (0,1,2) for x in (False,True)]
    assert table==expected
    A.pilot(mathcan,out/"lrf03-regression")
    controllers={m:C.Controller(mathcan,m) for m in C.MODES}
    base={"observation":1,"action":True,"root":C.ROOT,"actions":["unpack","enumerate"]}
    cases=[
        ("permit","compatible",{},"verified_permission",2),
        ("incomplete","compatible",{"actions":["skip"]},"certificate_denied",1),
        ("malformed_action","compatible",{"actions":["arbitrary Lean"]},"malformed_actions",0),
        ("wrong_root","compatible",{"root":"factor_of_section"},"wrong_required_theorem",0),
        ("denied_action","compatible",{"action":False},"policy_denied",0),
        ("denied_observation","compatible",{"observation":0},"policy_denied",0),
        ("unknown","compatible",{"observation":2},"unknown_observation",0),
        ("collision","collision",{},"authorization_collision",0),
        ("caller_receipt","compatible",{"receipt":{}},"malformed_request",0),
        ("malformed_observation","compatible",{"observation":True},"malformed_request",0),
    ]
    rows=[]
    for name,mode,change,reason,calls in cases:
        request={**base,**change}
        result=controllers[mode].review(request,out/"requests"/name)
        expected_allowed=reason=="verified_permission"
        assert (result["allowed"],result["reason"],result["compiler_calls"])==(expected_allowed,reason,calls),(name,result)
        rows.append({"case":name,"request":request,"result":result})
        print("AUTH_CASE",name,result["allowed"],result["reason"],result["compiler_calls"],flush=True)
    report={"passed":True,"version":version,"source_commit":A.COMMIT,"native_reference":native,
            "native_table":table,"table_cells":len(table),"request_cases":len(rows),
            "allows":sum(x["result"]["allowed"] for x in rows),
            "unauthorized_allows":0,"request_compiler_calls":sum(x["result"]["compiler_calls"] for x in rows),
            "lrf03_regression":json.loads((out/"lrf03-regression"/"gate.json").read_text())}
    (out/"requests.json").write_text(json.dumps(rows,indent=2)+"\n")
    (out/"gate.json").write_text(json.dumps(report,indent=2)+"\n")
    print("LRF04_GATE",json.dumps(report),flush=True)

if __name__=="__main__":main()
