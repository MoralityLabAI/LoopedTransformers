import {audit, perturbAudit, pilot} from "./kite01.mjs";
console.log(JSON.stringify({exact:audit(),perturbationChecks:perturbAudit(),...pilot()},null,2));
