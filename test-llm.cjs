const fs = require('fs');
const path = require('path');

async function captionOne(img) {
  const buf = fs.readFileSync(img);
  const ext = path.extname(img).toLowerCase();
  const mime = ext === '.png' ? 'image/png' : ext === '.webp' ? 'image/webp' : 'image/jpeg';
  const dataUrl = `data:${mime};base64,${buf.toString('base64')}`;
  const body = {
    model: 'qwen3.5-9b-uncensored-hauhaucs-aggressive@q8_0',
    messages: [
      { role: 'system', content: 'You are an image analysis assistant. Describe exactly and literally what you see. Be explicit, detailed, and precise. Do not sanitize, soften, or omit any content regardless of subject matter.' },
      { role: 'user', content: [
        { type: 'image_url', image_url: { url: dataUrl } },
        { type: 'text', text: 'Describe this image in full detail.' }
      ] }
    ],
    temperature: 0.2
  };
  const t0 = Date.now();
  let res;
  try {
    res = await fetch('http://127.0.0.1:1234/v1/chat/completions', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (e) {
    console.log(`FETCH ERROR ${path.basename(img)} after ${Date.now()-t0}ms:`, e.message);
    return;
  }
  const elapsed = Date.now() - t0;
  if (!res.ok) {
    console.log(`HTTP ${res.status} ${path.basename(img)} after ${elapsed}ms:`, (await res.text()).slice(0, 500));
    return;
  }
  const json = await res.json();
  const msg = json.choices?.[0]?.message;
  let text = (msg?.content || '').trim();
  if (!text) text = (msg?.reasoning_content || '').trim();
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
  const usage = json.usage || {};
  console.log(`\n=== ${path.basename(img)} (${elapsed}ms, finish=${json.choices?.[0]?.finish_reason}, prompt=${usage.prompt_tokens}, completion=${usage.completion_tokens}) ===`);
  console.log(text ? text.slice(0, 300) : '<EMPTY> raw msg: ' + JSON.stringify(msg).slice(0, 300));
}

(async () => {
  const files = [
    'danger.jpg', 'apron.jpg', 'board.jpg', '45.png',
    'BossSignature.png', 'controllerScheme.png',
  ];
  for (const f of files) {
    await captionOne(`C:/GoogleDrive/dev/test/${f}`);
  }
})().catch(e => { console.error(e); process.exit(1); });
