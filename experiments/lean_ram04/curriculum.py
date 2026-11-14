"""Statement-only authored curricula, with complete held-out template groups."""
import json
from pathlib import Path

TEMPLATES={
 'exact_direct':('exact_local',False,'(P Q : Prop) (aa : P) (bb : Q) : P'),
 'exact_application':('exact_local',False,'(P Q : Prop) (aa : P → Q) (bb : P) (cc : Q → P) : Q'),
 'exact_curried':('exact_local',False,'(P Q R : Prop) (aa : P → Q → R) (bb : P) (cc : Q) : R'),
 'exact_specialize':('exact_local',False,'(P : Nat → Prop) (nn : Nat) (aa : ∀ x, P x) (bb : nn = nn) : P nn'),
 'exact_symmetry':('exact_local',False,'{α : Type} (xx yy : α) (aa : xx = yy) : yy = xx'),
 'exact_projection':('exact_local',False,'(P Q R : Prop) (aa : P ∧ Q) (bb : R) : P'),
 'exact_dependent':('exact_local',True,'(P : Nat → Prop) (nn mm : Nat) (aa : ∀ x, P x → P (x+1)) (bb : P nn) : P (nn+1)'),
 'exact_quantified':('exact_local',True,'(P Q : Nat → Prop) (aa : ∀ x, P x) (bb : ∀ x, Q x) : ∀ y, P y'),
 'rw_unary':('rw_local',False,'{α : Type} (ff : α → Nat) (xx yy : α) (aa : xx = yy) : ff xx = ff yy'),
 'rw_binary':('rw_local',False,'{α : Type} (ff : α → α → Nat) (xx yy zz : α) (aa : xx = yy) : ff xx zz = ff yy zz'),
 'rw_chain':('rw_local',False,'{α : Type} (ff : α → Nat) (xx yy zz : α) (aa : xx = yy) (bb : yy = zz) : ff xx = ff zz'),
 'rw_hypothesis':('rw_local',False,'{α : Type} (P : α → Prop) (xx yy : α) (aa : xx = yy) (bb : P xx) : P yy'),
 'rw_nested':('rw_local',True,'{α : Type} (ff gg : α → α) (xx yy : α) (aa : xx = yy) : ff (gg xx) = ff (gg yy)'),
 'rw_iff':('rw_local',True,'(P Q R : Prop) (aa : P ↔ Q) (bb : P) (cc : R) : Q'),
 'rcases_and':('rcases_local',False,'(P Q : Prop) (aa : P ∧ Q) : Q ∧ P'),
 'rcases_exists':('rcases_local',False,'(P : Nat → Prop) (aa : ∃ x, P x) : True'),
 'rcases_or':('rcases_local',False,'(P Q : Prop) (aa : P ∨ Q) : Q ∨ P'),
 'rcases_nested':('rcases_local',False,'(P Q R : Prop) (aa : (P ∧ Q) ∧ R) : R ∧ P'),
 'rcases_exists_and':('rcases_local',True,'(P Q : Nat → Prop) (aa : ∃ x, P x ∧ Q x) : ∃ x, Q x ∧ P x'),
 'rcases_or_and':('rcases_local',True,'(P Q R : Prop) (aa : (P ∧ Q) ∨ (P ∧ R)) : P'),
}

def generate(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);roots=[]
    for template,(family,dev,statement) in TEMPLATES.items():
        for variant in range(4 if dev else 2):
            # Rotate names across roles; never expose template IDs as model features.
            names=['h','k','cert','given'];offset=variant%4
            renamed=statement
            import re
            mapping={word:names[(i+offset)%4] for i,word in enumerate(('aa','bb','cc'))}
            mapping.update(nn='n'+str(variant),mm='m'+str(variant),xx='x'+str(variant),yy='y'+str(variant),zz='z'+str(variant))
            renamed=re.sub(r'\b(?:aa|bb|cc|nn|mm|xx|yy|zz)\b',lambda m:mapping[m[0]],renamed)
            name=template+f'_{variant}';filename=name+'.lean'
            source=f'import Init\nnamespace RAM04Curriculum\ntheorem {name} {renamed} := by\n  sorry\nend RAM04Curriculum\n'
            (out/filename).write_text(source,encoding='utf8',newline='\n')
            roots.append(dict(source_file='Curriculum/'+filename,theorem=name,origin='authored',template=template,
                              target_family=family,development=dev))
    (out/'roots.json').write_text(json.dumps(roots,indent=2)+'\n',encoding='utf8',newline='\n')
    return roots
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(len(generate(a.out)))
