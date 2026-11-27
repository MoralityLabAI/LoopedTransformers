import {writeFileSync} from 'node:fs';
import {pilot03} from './joint03_pilot.mjs';
const r=await pilot03();if(process.argv[2])writeFileSync(process.argv[2],JSON.stringify(r)+'\n');console.log(JSON.stringify({status:r.status,roots:r.roots,summary:r.summary,paired:r.paired}));
