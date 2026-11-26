import {readFileSync} from 'node:fs';
import {evaluate03} from './router03.mjs';
console.log(JSON.stringify(evaluate03(...process.argv.slice(2,4).map(p=>JSON.parse(readFileSync(p,'utf8')))),null,2));
