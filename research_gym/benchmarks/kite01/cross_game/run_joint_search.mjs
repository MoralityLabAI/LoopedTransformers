import {writeFileSync} from 'node:fs';
import {boundsTest,pilot} from './joint_search_pilot.mjs';
const bounds=await boundsTest(),r=await pilot();
if(process.argv[2])writeFileSync(process.argv[2],JSON.stringify({...r,bounds},null,2)+'\n');
console.log(JSON.stringify({bounds,summary:r.summary,paired:r.paired},null,2));
