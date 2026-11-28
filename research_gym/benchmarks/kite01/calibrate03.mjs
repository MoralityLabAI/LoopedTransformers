import {readFileSync} from 'node:fs';
import {calibration03} from './router03.mjs';
console.log(JSON.stringify(calibration03(JSON.parse(readFileSync(process.argv[2],'utf8'))),null,2));
