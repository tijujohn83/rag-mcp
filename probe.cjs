const Database = require('better-sqlite3');
const db = new Database('rag-index.db');
const rows = db.prepare("SELECT file, substr(text,1,400) AS snippet FROM chunks WHERE file LIKE '%/test/%' AND text LIKE '[LLM_CAPTION]%' ORDER BY file").all();
console.log('Captioned chunks:', rows.length, '\n');
for (const r of rows) {
  console.log('===', r.file.replace('C:/GoogleDrive/dev/test/', ''));
  console.log(r.snippet.replace(/\s+/g, ' '));
  console.log();
}
const all = db.prepare("SELECT file, count(*) AS c FROM chunks WHERE file LIKE '%/test/%' GROUP BY file ORDER BY file").all();
console.log('Per-file chunk counts:');
for (const r of all) console.log(' ', r.c.toString().padStart(4), r.file.replace('C:/GoogleDrive/dev/test/', ''));
