import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import {audit, perturbAudit, pilot} from './kite01.mjs';
import {hardPilot} from './hard.mjs';
import {evaluate} from './router02.mjs';
import {evaluate03} from './router03.mjs';

const json = p => JSON.parse(readFileSync(new URL(p,import.meta.url),'utf8'));
const stable = (a,b) => assert.deepStrictEqual(a,b);
stable({exact:audit(),perturbationChecks:perturbAudit(),...pilot()},json('./development_receipt.json'));
stable(hardPilot(),json('./hard_receipt.json'));
const r2=evaluate(json('./router02_checkpoint.json'));
for(const arm of new Set(r2.rows.map(r=>r.arm)))stable(r2.rows.filter(r=>r.arm===arm),json('./router02_receipts/'+arm+'.json').rows);
for(const {seed,rows} of r2.sensitivity)stable(rows,json('./router02_receipts/learned_seed_'+seed+'.json').rows);
const r3=evaluate03(json('./kite03_checkpoint.json'),json('./kite03_dev.json'));
for(const arm of new Set(r3.rows.map(r=>r.arm)))stable(r3.rows.filter(r=>r.arm===arm),json('./kite03_receipts/'+arm+'.json').rows);
for(const {seed,rows} of r3.sensitivity)stable(rows,json('./kite03_receipts/learned_seed_'+seed+'.json').rows);
console.log(JSON.stringify({status:'PASS',kite01_profiles:1280,kite01b_profiles:960,kite02_profiles:r2.rows.length+r2.sensitivity.reduce((s,m)=>s+m.rows.length,0),kite03_profiles:r3.rows.length+r3.sensitivity.reduce((s,m)=>s+m.rows.length,0)}));
