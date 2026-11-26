import {readFileSync} from 'node:fs';
import {evaluate} from './router02.mjs';
console.log(JSON.stringify(evaluate(JSON.parse(readFileSync(process.argv[2],'utf8'))),null,2));
